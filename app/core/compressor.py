"""Compression engine — selectively prunes low-surprisal tokens."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Any

from app.core.model_loader import ModelLoader
from app.utils.token_utils import compute_threshold, rebuild_text_from_tokens

logger = logging.getLogger(__name__)


FLUFF_PATTERNS = [
    r"\bplease\b",
    r"\bthank\s+you\b",
    r"\bcould\s+you\b",
    r"\bcan\s+you\s+help\s+me\b",
    r"\bcan\s+you\b",
    r"\bi\s+would\s+like\s+to\b",
    r"\bsorry\s+but\b",
    r"\bact\s+as\s+a\b",
    r"\bwould\s+you\b",
    r"\bkindly\b",
]


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
        kept_indices: List[int] = []
        removed_tokens: List[str] = []

        for idx, (tok, score) in enumerate(zip(token_strings, surprisal_scores)):
            # Always keep the first token and tokens that are purely whitespace
            # or punctuation — removing them usually destroys readability.
            if idx == 0 or score >= threshold or self._is_structural(tok):
                kept_tokens.append(tok)
                kept_indices.append(idx)
            else:
                removed_tokens.append(tok)

        # 4. Reconstruct text
        compressed_text = rebuild_text_from_tokens(kept_tokens)
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
        # Keep common structural symbols
        if all(ch in ".,;:!?()[]{}\"'-/\\@#$%^&*+=<>|~`\n\r\t" for ch in cleaned):
            return True
        return False
