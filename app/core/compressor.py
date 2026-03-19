"""Compression engine — selectively prunes low-surprisal tokens."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Any

from app.core.model_loader import ModelLoader
from app.utils.token_utils import compute_threshold

logger = logging.getLogger(__name__)


FLUFF_PATTERNS = [
    r"\bplease\b",
    r"\bthank\s+you\b",
    r"\bcould\s+you\b",
    r"\bcan\s+you\s+help\s+me\b",
    r"\bcan\s+you\b",
    r"\bi\s+would\s+like\s+to\b",
    r"\bi\s+need\s+you\s+to\b",
    r"\bi\s+want\s+you\s+to\b",
    r"\bsorry\s+but\b",
    r"\bsorry\b",
    r"\bact\s+as\s+a\b",
    r"\bwould\s+you\b",
    r"\bkindly\b",
    r"\bjust\b",
    r"\bif\s+possible\b",
    r"\bactually\b",
    r"\bbasically\b",
    r"\bit\s+would\s+be\s+great\s+if\b",
    r"\bi\s+was\s+wondering\s+if\b",
    r"\bi\s+am\s+looking\s+for\b",
    r"\bdo\s+you\s+think\s+you\s+could\b",
    r"\bi\s+would\s+appreciate\s+it\s+if\b",
    r"\bif\s+you\s+don'?t\s+mind\b",
    r"\bwould\s+it\s+be\s+possible\s+to\b",
]


# Structural simplification rules — ordered regex substitutions.
STRUCTURE_RULES: List[tuple] = [
    (r"can\s+you\s+(?:please\s+)?explain\s+(?:to\s+me\s+)?how\s+(?:I\s+can|to)\s+", ""),
    (r"can\s+you\s+(?:please\s+)?(?:help\s+me\s+)?(?:to\s+)?", ""),
    (r"please\s+(?:explain|describe|show)\s+(?:to\s+me\s+)?(?:how\s+(?:to|I\s+can)\s+)?", ""),
    (r"how\s+(?:can|do)\s+I\s+", ""),
    (r"I\s+(?:want|need)\s+(?:to|you\s+to)\s+", ""),
    (r"could\s+you\s+(?:please\s+)?(?:help\s+me\s+)?", ""),
    (r"would\s+you\s+(?:be\s+able\s+to\s+)?", ""),
    (r"is\s+it\s+possible\s+(?:to|for\s+you\s+to)\s+", ""),
    (r"I\s+would\s+like\s+(?:to|you\s+to)\s+", ""),
    (r"I\s+am\s+trying\s+to\s+", ""),
    (r"what\s+is\s+the\s+(?:best\s+)?way\s+to\s+", ""),
]


def structurally_simplify(text: str) -> str:
    """Convert verbose natural language into compact instruction form.

    Example::

        >>> structurally_simplify("can you please explain how I can write a python function")
        "write a python function"
    """
    result = text
    for pattern, replacement in STRUCTURE_RULES:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    result = re.sub(r"\s{2,}", " ", result).strip()
    return result if result else text


def clean_prompt_text(text: str) -> str:
    """Remove conversational filler while preserving instruction semantics."""
    cleaned = text or ""

    for pattern in FLUFF_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

    # Remove repeated punctuation/spaces left behind by phrase stripping.
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = cleaned.strip(" ,;:-\n\t")

    # Safety fallback: avoid returning an empty prompt after cleanup.
    return cleaned if cleaned else (text or "").strip()


@dataclass
class CompressionResult:
    """Structured output of the compression pipeline."""

    compressed_text: str
    removed_tokens: List[str]
    original_token_count: int
    compressed_token_count: int
    token_reduction_percent: float
    kept_indices: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compressed_text": self.compressed_text,
            "removed_tokens": self.removed_tokens,
            "original_token_count": self.original_token_count,
            "compressed_token_count": self.compressed_token_count,
            "token_reduction_percent": round(self.token_reduction_percent, 4),
        }


class Compressor:
    """Token-level prompt compressor driven by surprisal scoring.

    Low-surprisal tokens carry less information and are candidates for
    removal.  The *aggressiveness* parameter (0.0 - 1.0) controls how
    aggressively the engine prunes.
    """

    def __init__(self, model_loader: ModelLoader) -> None:
        self._model = model_loader

    def compress_prompt(
        self,
        text: str,
        aggressiveness: float = 0.3,
    ) -> CompressionResult:
        """Compress *text* by removing low-information tokens.

        Args:
            text: the original prompt.
            aggressiveness: pruning intensity in [0.0, 1.0].

        Returns:
            A ``CompressionResult`` with the compressed text, removed tokens,
            and reduction metrics.
        """
        aggressiveness = max(0.0, min(1.0, aggressiveness))

        preprocessed_text = clean_prompt_text(text)

        # 1. Tokenize & score
        input_ids, token_strings = self._model.tokenize(preprocessed_text)
        surprisal_scores: List[float] = self._model.compute_token_surprisal(preprocessed_text)

        original_count = len(token_strings)

        # 2. Determine threshold
        threshold = compute_threshold(surprisal_scores, aggressiveness)

        # 3. Partition tokens
        kept_tokens: List[str] = []
        kept_token_ids: List[int] = []
        kept_indices: List[int] = []
        removed_tokens: List[str] = []

        for idx, (tok, score) in enumerate(zip(token_strings, surprisal_scores)):
            # Always keep the first token and tokens that are purely whitespace
            # or punctuation — removing them usually destroys readability.
            if idx == 0 or score >= threshold or self._is_structural(tok):
                kept_tokens.append(tok)
                kept_token_ids.append(int(input_ids[0, idx].item()))
                kept_indices.append(idx)
            else:
                removed_tokens.append(tok)

        # 4. Reconstruct text
        # Decode from token ids for correct byte-level token handling.
        compressed_text = self._model.decode_tokens(kept_token_ids)
        compressed_count = len(kept_tokens)

        reduction_pct = (
            ((original_count - compressed_count) / original_count) * 100.0
            if original_count > 0
            else 0.0
        )

        logger.info(
            "Compressed %d → %d tokens (%.1f%% reduction, aggressiveness=%.2f)",
            original_count,
            compressed_count,
            reduction_pct,
            aggressiveness,
        )

        return CompressionResult(
            compressed_text=compressed_text,
            removed_tokens=removed_tokens,
            original_token_count=original_count,
            compressed_token_count=compressed_count,
            token_reduction_percent=reduction_pct,
            kept_indices=kept_indices,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_structural(token: str) -> bool:
        """Return True if *token* is punctuation / whitespace that should be kept."""
        cleaned = token.replace("Ġ", "").replace("▁", "").strip()
        if not cleaned:
            return True
        # Byte-level BPE may split Unicode punctuation into non-ASCII
        # fragments; dropping those creates replacement-char artifacts.
        if any(ord(ch) > 127 for ch in cleaned):
            return True
        # Keep common structural symbols
        if all(ch in ".,;:!?()[]{}\"'-/\\@#$%^&*+=<>|~`\n\r\t" for ch in cleaned):
            return True
        return False


def compress_with_genome(text: str, genome_rules: List[str], use_cache: bool = True) -> str:
    """Apply an evolved ordered rule genome to a prompt."""
    if not text or not genome_rules:
        return text

    try:
        # Imported lazily to keep the main app functional if evolution modules
        # are not present in a minimal deployment.
        from rules import apply_rule

        out = text
        for rule in genome_rules:
            out = apply_rule(rule, out, use_cache=use_cache)
        return " ".join(out.split())
    except Exception:
        logger.exception("Genome compression failed; returning original prompt")
        return text


def fluency_penalty(original: str, compressed: str) -> float:
    """Heuristic fluency/structure penalty used in shared fitness scoring."""
    penalty = 0.0
    if original.count("\n") >= 2 and compressed.count("\n") == 0:
        penalty += 0.15
    if len(compressed.strip()) < 40:
        penalty += 0.25
    if compressed.count("(") != compressed.count(")"):
        penalty += 0.10
    if compressed.count("[") != compressed.count("]"):
        penalty += 0.08
    return min(1.0, penalty)


def score_compression_variant(
    original_prompt: str,
    candidate_prompt: str,
    evaluator: "Evaluator",  # noqa: F821
    density_metrics: "DensityMetrics",  # noqa: F821
    test_query: str,
) -> Dict[str, float]:
    """Compute reusable fitness metrics for a candidate compression."""
    density_report = density_metrics.score(original_prompt, candidate_prompt)
    eval_report = evaluator.evaluate(
        original_prompt=original_prompt,
        compressed_prompt=candidate_prompt,
        test_query=test_query,
    )

    reduction = max(0.0, min(1.0, 1.0 - density_report.compression_ratio))
    drift = max(0.0, min(1.0, eval_report.drift_score))
    penalty = fluency_penalty(original_prompt, candidate_prompt)

    # Match aggressive compression weighting from evolution fitness.
    fitness = (2.5 * reduction) - (0.6 * drift) - (0.4 * penalty)
    fitness = math.tanh(fitness)

    return {
        "reduction_percent": reduction * 100.0,
        "drift_score": drift,
        "fluency_penalty": penalty,
        "fitness": fitness,
    }
