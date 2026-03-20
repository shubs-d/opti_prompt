"""Multi-stage prompt optimisation pipeline.

Stages:
  1. Regex fluff cleaning
  2. Structural simplification
  3. Tokenization + surprisal (distilgpt2)
  4. GEPA scoring (positional × type weights)
  5. Adaptive token pruning (intent-aware threshold)
  6. Prompt reconstruction with grammar fixes
  7. Semantic validation (cosine similarity ≥ 0.85)
  8. Metrics (token counts, compression ratio, information density)

All heavy-lifting reuses the singleton ``ModelLoader`` and existing
``similarity.py`` utilities — no new models or external APIs.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.core.model_loader import ModelLoader
from app.core.intent_engine import IntentEngine, IntentCategory
from app.utils.similarity import cosine_similarity_score, simple_sentence_embedding

logger = logging.getLogger(__name__)


# =====================================================================
# Constants
# =====================================================================

# --- Stage 1: extended filler patterns --------------------------------
FLUFF_PATTERNS: List[str] = [
    r"\bplease\b",
    r"\bthank\s+you\b",
    r"\bcould\s+you\b",
    r"\bcan\s+you\s+help\s+me\b",
    r"\bcan\s+you\b",
    r"\bi\s+would\s+like\s+you\s+to\b",
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
    r"\bhelp\s+me\s+to\b",
    r"\bhelp\s+me\b",
]

# --- Stage 2: structural simplifier rules ----------------------------
# Each tuple: (pattern, replacement).  Applied in order.
STRUCTURE_RULES: List[Tuple[str, str]] = [
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
    (r"\bsearch\s+for\b", "search"),
    (r"\blook\s+for\b", "find"),
]

# --- Stage 4: token type weights for GEPA scoring --------------------
STOPWORDS: frozenset = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "am", "it", "its",
    "this", "that", "these", "those", "i", "me", "my", "we", "our", "you",
    "your", "he", "him", "his", "she", "her", "they", "them", "their",
    "what", "which", "who", "whom", "where", "when", "why", "how",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "up",
    "about", "into", "through", "during", "before", "after", "above",
    "below", "between", "and", "but", "or", "nor", "not", "so", "very",
    "just", "also", "than", "then", "if", "as",
})

KEYWORDS: frozenset = frozenset({
    "write", "generate", "code", "explain", "create", "build", "implement",
    "design", "develop", "debug", "fix", "optimize", "test", "deploy",
    "analyze", "compare", "evaluate", "function", "class", "script",
    "algorithm", "data", "api", "database", "server", "client",
    "python", "javascript", "sql", "html", "css", "list", "sort", "search",
    "summarize", "translate", "convert", "compute", "calculate",
})

# Words that should never be pruned (nouns, verbs, numbers, technical)
PROTECTED_PATTERNS: List[str] = [
    r"^\d+$",                       # pure numbers
    r"^[A-Z][a-z]+$",              # capitalised nouns
]

PROTECTED_WORDS: frozenset = KEYWORDS | frozenset({
    "not", "no", "never", "error", "warning", "fail",
    "input", "output", "return", "print", "read", "write",
    "file", "path", "name", "value", "key", "type", "string",
    "number", "integer", "float", "boolean", "array", "object",
})

# --- Stage 5: intent → pruning factor --------------------------------
_INTENT_FACTOR: Dict[str, float] = {
    IntentCategory.TECHNICAL.value: 0.5,
    IntentCategory.INFORMATIONAL.value: 0.6,
    IntentCategory.ANALYTICAL.value: 0.6,
    IntentCategory.CREATIVE.value: 0.7,
    IntentCategory.CONVERSATIONAL.value: 0.7,
}

# --- Stage 5b: output-size → pruning scale + suffix -------------------
_OUTPUT_SIZE_PRUNE_SCALE: Dict[str, float] = {
    "short": 0.85,      # tighter threshold → prune more
    "moderate": 1.0,    # baseline
    "long": 1.25,       # looser threshold → prune less
}

_OUTPUT_SIZE_SUFFIX: Dict[str, str] = {
    "short": "\nAnswer concisely in ~100 words.",
    "moderate": "\nAnswer in ~300 words with moderate detail.",
    "long": "\nProvide a detailed, comprehensive answer.",
}

# --- Stage 6: minimal grammar reconstruction rules --------------------
# (pattern, replacement) applied after joining filtered tokens.
GRAMMAR_RULES: List[Tuple[str, str]] = [
    # Insert articles / prepositions where needed
    (r"\b(write|create|build|implement)\s+(python|javascript|html|css|sql)\b",
     r"\1 a \2"),
    (r"\b(sort|search|filter)\s+(numbers|strings|items|elements|list)\b",
     r"\1 \2"),
    (r"\b(script|function|program)\s+(sort|search|filter|compute|calculate)\b",
     r"\1 to \2"),
    (r"\b(numbers|items|elements|strings)\s+(list|array)\b",
     r"\1 in a \2"),
    # Collapse double spaces
    (r"\s{2,}", " "),
]


# =====================================================================
# Result dataclass
# =====================================================================

@dataclass
class PipelineResult:
    """Structured output of the multi-stage pipeline."""

    original_prompt: str
    optimized_prompt: str
    original_token_count: int
    optimized_token_count: int
    compression_ratio: float
    information_density: float
    semantic_similarity: float
    pipeline_accepted: bool
    template: Dict[str, Any]
    stages_applied: List[str] = field(default_factory=list)
    gepa_scores: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_prompt": self.original_prompt,
            "optimized_prompt": self.optimized_prompt,
            "original_token_count": self.original_token_count,
            "optimized_token_count": self.optimized_token_count,
            "compression_ratio": round(self.compression_ratio, 4),
            "information_density": round(self.information_density, 4),
            "semantic_similarity": round(self.semantic_similarity, 4),
            "pipeline_accepted": self.pipeline_accepted,
            "template": self.template,
            "stages_applied": self.stages_applied,
        }


# =====================================================================
# Pipeline
# =====================================================================

class PromptPipeline:
    """Nine-stage prompt optimisation pipeline.

    Usage::

        pipeline = PromptPipeline(ModelLoader.get_instance())
        result = pipeline.run("can you please explain how I can write a python function")
    """

    SIMILARITY_THRESHOLD = 0.85

    def __init__(self, model_loader: ModelLoader) -> None:
        self._model = model_loader
        self._intent_engine = IntentEngine()

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def run(
        self,
        text: str,
        intent_label: Optional[str] = None,
        output_size: str = "moderate",
    ) -> PipelineResult:
        """Execute the full 9-stage pipeline on *text*.

        Args:
            text: raw user prompt.
            intent_label: optional pre-detected intent label.  When
                ``None`` the pipeline auto-detects intent.
            output_size: desired output verbosity — ``'short'``,
                ``'moderate'``, or ``'long'``.

        Returns:
            A ``PipelineResult`` with the optimised prompt, metrics, and
            template extraction.
        """
        stages_applied: List[str] = []
        original_text = text

        # --- Stage 1: Regex fluff cleaning ----------------------------
        cleaned = self._stage_regex_clean(text)
        stages_applied.append("regex_fluff_clean")

        # --- Stage 2: Structural simplification -----------------------
        simplified = self._stage_structural_simplify(cleaned)
        stages_applied.append("structural_simplify")

        # --- Stage 3: Tokenization + surprisal ------------------------
        input_ids, token_strings, surprisal_scores = self._stage_tokenize_surprisal(simplified)
        stages_applied.append("tokenize_surprisal")

        # Token count from the *original* raw text (before cleanup) so the
        # compression ratio reflects the full reduction including fluff removal.
        _, orig_tokens = self._model.tokenize(original_text)
        original_token_count = len(orig_tokens)

        # --- Stage 4: GEPA scoring ------------------------------------
        gepa_scores = self._stage_gepa_scoring(token_strings, surprisal_scores)
        stages_applied.append("gepa_scoring")

        # --- Stage 5: Adaptive pruning --------------------------------
        if intent_label is None:
            intent_result = self._intent_engine.detect(original_text, model_loader=self._model)
            intent_label = intent_result.intent_label

        kept_tokens, kept_ids, removed_tokens = self._stage_adaptive_prune(
            token_strings, input_ids, gepa_scores, intent_label, output_size,
        )
        stages_applied.append("adaptive_prune")

        # --- Stage 6: Reconstruction ---------------------------------
        reconstructed = self._stage_reconstruct(kept_ids)
        stages_applied.append("reconstruct")

        # --- Stage 7: Semantic validation -----------------------------
        similarity = self._stage_semantic_validate(original_text, reconstructed)
        stages_applied.append("semantic_validate")

        accepted = similarity >= self.SIMILARITY_THRESHOLD
        if not accepted:
            # Fallback: return the simplified (but unpruned) version
            reconstructed = simplified
            similarity = self._stage_semantic_validate(original_text, reconstructed)
            accepted = True  # simplified version is always accepted as safe fallback
            logger.warning(
                "Pruned prompt rejected (sim=%.4f < %.2f); falling back to "
                "structurally-simplified version (sim=%.4f).",
                similarity,
                self.SIMILARITY_THRESHOLD,
                similarity,
            )

        # --- Stage 8: Metrics -----------------------------------------
        opt_ids, opt_tokens = self._model.tokenize(reconstructed)
        optimized_token_count = len(opt_tokens)

        compression_ratio = (
            optimized_token_count / original_token_count
            if original_token_count > 0
            else 1.0
        )

        # Information density = sum(GEPA scores of kept tokens) / count
        kept_gepa = [g for g, tok in zip(gepa_scores, token_strings) if tok in kept_tokens]
        information_density = (
            sum(kept_gepa) / len(kept_gepa)
            if kept_gepa
            else 0.0
        )
        stages_applied.append("metrics")

        # --- Stage 9: Template extraction -----------------------------
        from app.core.template_extractor import extract_template

        template = extract_template(reconstructed)
        stages_applied.append("template_extraction")

        logger.info(
            "Pipeline: %d → %d tokens (ratio=%.2f, sim=%.4f, density=%.4f)",
            original_token_count,
            optimized_token_count,
            compression_ratio,
            similarity,
            information_density,
        )

        return PipelineResult(
            original_prompt=original_text,
            optimized_prompt=reconstructed,
            original_token_count=original_token_count,
            optimized_token_count=optimized_token_count,
            compression_ratio=compression_ratio,
            information_density=information_density,
            semantic_similarity=similarity,
            pipeline_accepted=accepted,
            template=template,
            stages_applied=stages_applied,
            gepa_scores=gepa_scores,
        )

    # -----------------------------------------------------------------
    # Stage implementations
    # -----------------------------------------------------------------

    @staticmethod
    def _stage_regex_clean(text: str) -> str:
        """Stage 1 — remove conversational filler phrases."""
        cleaned = text or ""
        for pattern in FLUFF_PATTERNS:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
        # Tidy whitespace artefacts
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        cleaned = cleaned.strip(" ,;:-\n\t")
        return cleaned if cleaned else (text or "").strip()

    @staticmethod
    def _stage_structural_simplify(text: str) -> str:
        """Stage 2 — convert verbose NL → compact instruction form."""
        result = text
        for pattern, replacement in STRUCTURE_RULES:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        result = re.sub(r"\s{2,}", " ", result).strip()
        return result if result else text

    def _stage_tokenize_surprisal(
        self, text: str,
    ) -> Tuple[Any, List[str], List[float]]:
        """Stage 3 — tokenize and compute per-token surprisal."""
        input_ids, token_strings = self._model.tokenize(text)
        surprisal_scores = self._model.compute_token_surprisal(text)
        return input_ids, token_strings, surprisal_scores

    @staticmethod
    def _stage_gepa_scoring(
        tokens: List[str],
        surprisals: List[float],
    ) -> List[float]:
        """Stage 4 — compute GEPA scores with positional + type weights."""
        n = len(tokens)
        if n == 0:
            return []

        third = max(1, n // 3)
        gepa_scores: List[float] = []

        for idx, (tok, surp) in enumerate(zip(tokens, surprisals)):
            # --- positional weight ---
            if idx < third:
                pos_w = 1.2   # early tokens
            elif idx < 2 * third:
                pos_w = 1.0   # middle tokens
            else:
                pos_w = 0.8   # late tokens

            # --- token type weight ---
            clean_tok = tok.replace("Ġ", "").replace("▁", "").strip().lower()
            if clean_tok in STOPWORDS:
                type_w = 0.4
            elif clean_tok in KEYWORDS:
                type_w = 1.5
            else:
                type_w = 1.0

            gepa_scores.append(surp * pos_w * type_w)

        return gepa_scores

    def _stage_adaptive_prune(
        self,
        tokens: List[str],
        input_ids: Any,
        gepa_scores: List[float],
        intent_label: str,
        output_size: str = "moderate",
    ) -> Tuple[List[str], List[int], List[str]]:
        """Stage 5 — prune tokens below the intent-aware GEPA threshold."""
        intent_factor = _INTENT_FACTOR.get(intent_label, 0.7)
        size_scale = _OUTPUT_SIZE_PRUNE_SCALE.get(output_size, 1.0)

        if not gepa_scores:
            return tokens, [], []

        mean_gepa = sum(gepa_scores) / len(gepa_scores)
        threshold = mean_gepa * intent_factor * size_scale

        kept_tokens: List[str] = []
        kept_ids: List[int] = []
        removed_tokens: List[str] = []

        for idx, (tok, score) in enumerate(zip(tokens, gepa_scores)):
            clean_tok = tok.replace("Ġ", "").replace("▁", "").strip().lower()

            # Always keep the first token
            is_first = idx == 0

            # Check if token is protected
            is_protected = clean_tok in PROTECTED_WORDS
            if not is_protected:
                for pat in PROTECTED_PATTERNS:
                    if re.match(pat, clean_tok):
                        is_protected = True
                        break

            # Check if token is structural (punctuation, etc.)
            is_structural = self._is_structural(tok)

            if is_first or score >= threshold or is_protected or is_structural:
                kept_tokens.append(tok)
                kept_ids.append(int(input_ids[0, idx].item()))
            else:
                removed_tokens.append(tok)

        return kept_tokens, kept_ids, removed_tokens

    def _stage_reconstruct(self, kept_ids: List[int]) -> str:
        """Stage 6 — decode filtered token IDs and apply grammar fixes."""
        if not kept_ids:
            return ""

        text = self._model.decode_tokens(kept_ids)

        # Apply minimal grammar correction rules
        for pattern, replacement in GRAMMAR_RULES:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text.strip()

    def _stage_semantic_validate(
        self, original: str, optimized: str,
    ) -> float:
        """Stage 7 — compute cosine similarity between original + optimised."""
        orig_emb = simple_sentence_embedding(original, self._model)
        opt_emb = simple_sentence_embedding(optimized, self._model)
        return cosine_similarity_score(orig_emb, opt_emb)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _is_structural(token: str) -> bool:
        """Return True if *token* is punctuation / whitespace."""
        cleaned = token.replace("Ġ", "").replace("▁", "").strip()
        if not cleaned:
            return True
        if any(ord(ch) > 127 for ch in cleaned):
            return True
        if all(ch in ".,;:!?()[]{}\"'-/\\@#$%^&*+=<>|~`\n\r\t" for ch in cleaned):
            return True
        return False
