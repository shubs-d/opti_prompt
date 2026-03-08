"""Intent identification engine — classifies prompt intent and recommends aggressiveness.

Supports two detection strategies:
  1. **Rule-based** (default) — keyword / pattern matching, zero extra RAM.
  2. **Embedding-based** — cosine similarity of prompt embedding against
     intent-descriptor embeddings (reuses the already-loaded causal LM).

The strategy is selected via the ``INTENT_STRATEGY`` env variable
(``rule`` | ``embedding``).  Falls back to rule-based when the model
is unavailable.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Intent taxonomy
# ------------------------------------------------------------------

class IntentCategory(str, Enum):
    """High-level prompt intent categories."""

    INFORMATIONAL = "INFORMATIONAL"
    CREATIVE = "CREATIVE"
    TECHNICAL = "TECHNICAL"
    ANALYTICAL = "ANALYTICAL"
    CONVERSATIONAL = "CONVERSATIONAL"


# Aggressiveness ranges per intent  (min, default, max)
_AGGRESSIVENESS_MAP: Dict[IntentCategory, Tuple[float, float, float]] = {
    IntentCategory.INFORMATIONAL: (0.40, 0.50, 0.60),
    IntentCategory.CREATIVE:      (0.20, 0.30, 0.40),
    IntentCategory.TECHNICAL:     (0.10, 0.20, 0.30),
    IntentCategory.ANALYTICAL:    (0.30, 0.40, 0.50),
    IntentCategory.CONVERSATIONAL:(0.50, 0.60, 0.70),
}


# Per-intent optimisation strategy hints used by the candidate generator
# and decision engine to tailor the pipeline per intent type.
@dataclass
class IntentStrategy:
    """Optimisation strategy parameters for a given intent."""

    prefer_structuring: bool
    min_similarity: float          # floor below which we reject / fall back
    density_weight: float          # weight of density in composite score
    similarity_weight: float       # weight of similarity in composite score
    compression_weight: float      # weight of compression ratio in composite score
    skip_aggressive_candidate: bool  # whether to omit the aggressive candidate


_INTENT_STRATEGIES: Dict[IntentCategory, IntentStrategy] = {
    IntentCategory.INFORMATIONAL: IntentStrategy(
        prefer_structuring=True,
        min_similarity=0.85,
        density_weight=0.40,
        similarity_weight=0.40,
        compression_weight=0.20,
        skip_aggressive_candidate=False,
    ),
    IntentCategory.CREATIVE: IntentStrategy(
        prefer_structuring=False,
        min_similarity=0.92,
        density_weight=0.20,
        similarity_weight=0.60,
        compression_weight=0.20,
        skip_aggressive_candidate=True,
    ),
    IntentCategory.TECHNICAL: IntentStrategy(
        prefer_structuring=True,
        min_similarity=0.90,
        density_weight=0.35,
        similarity_weight=0.50,
        compression_weight=0.15,
        skip_aggressive_candidate=True,
    ),
    IntentCategory.ANALYTICAL: IntentStrategy(
        prefer_structuring=True,
        min_similarity=0.87,
        density_weight=0.35,
        similarity_weight=0.45,
        compression_weight=0.20,
        skip_aggressive_candidate=False,
    ),
    IntentCategory.CONVERSATIONAL: IntentStrategy(
        prefer_structuring=False,
        min_similarity=0.80,
        density_weight=0.30,
        similarity_weight=0.35,
        compression_weight=0.35,
        skip_aggressive_candidate=False,
    ),
}


def get_intent_strategy(intent_label: str) -> IntentStrategy:
    """Retrieve the ``IntentStrategy`` for a given intent label string."""
    try:
        cat = IntentCategory(intent_label.upper())
    except ValueError:
        cat = IntentCategory.INFORMATIONAL
    return _INTENT_STRATEGIES[cat]


@dataclass
class IntentResult:
    """Structured output from intent detection."""

    intent_label: str
    confidence_score: float
    recommended_aggressiveness: float
    strategy: str  # "rule" | "embedding"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent_label": self.intent_label,
            "confidence_score": round(self.confidence_score, 4),
            "recommended_aggressiveness": round(self.recommended_aggressiveness, 4),
            "strategy": self.strategy,
        }

    def get_optimization_strategy(self) -> "IntentStrategy":
        """Return the ``IntentStrategy`` associated with this result."""
        return get_intent_strategy(self.intent_label)


# ------------------------------------------------------------------
# Keyword banks for rule-based classification
# ------------------------------------------------------------------

_KEYWORD_BANKS: Dict[IntentCategory, List[str]] = {
    IntentCategory.INFORMATIONAL: [
        r"\bexplain\b", r"\bdefine\b", r"\bdefinition\b", r"\bwhat\s+is\b",
        r"\bwhat\s+are\b", r"\bdescribe\b", r"\bsummar", r"\boverview\b",
        r"\btell\s+me\s+about\b", r"\bfact\b", r"\bmeaning\b", r"\blist\b",
        r"\bhow\s+does\b", r"\bhow\s+do\b", r"\bwhy\s+does\b", r"\bwhy\s+do\b",
        r"\binformation\b", r"\bknowledge\b", r"\bwho\s+is\b", r"\bwhen\s+did\b",
    ],
    IntentCategory.CREATIVE: [
        r"\bwrite\b", r"\bstory\b", r"\bpoem\b", r"\bcreative\b",
        r"\bbrainstorm\b", r"\bimagine\b", r"\binvent\b", r"\bfiction\b",
        r"\bmarketing\b", r"\bslogan\b", r"\btagline\b", r"\bcatchy\b",
        r"\bnarrat", r"\bdraft\b", r"\bgenerate\b.*\bcontent\b",
        r"\bcopy\s*writ", r"\bad\s+copy\b", r"\bblog\s+post\b",
        r"\bscript\b", r"\bdialogue\b", r"\bsong\b",
    ],
    IntentCategory.TECHNICAL: [
        r"\bcode\b", r"\bfunction\b", r"\bclass\b", r"\bmethod\b",
        r"\bdebug\b", r"\berror\b", r"\bbug\b", r"\bfix\b",
        r"\barchitect", r"\bapi\b", r"\bsql\b", r"\bpython\b",
        r"\bjavascript\b", r"\btypescript\b", r"\bjava\b", r"\brust\b",
        r"\bimport\b", r"\balgorithm\b", r"\bdata\s*struct",
        r"\bimplement\b", r"\brefactor\b", r"\btest\b",
        r"\bdocker\b", r"\bkubernetes\b", r"\bgit\b", r"\bci/cd\b",
        r"\bcompil", r"\bsyntax\b", r"\bstack\s*trace\b",
    ],
    IntentCategory.ANALYTICAL: [
        r"\bcompar", r"\banalyz", r"\banalys", r"\bevaluat",
        r"\bpros?\s+and\s+cons?\b", r"\btrade\s*off", r"\badvantage",
        r"\bdisadvantage", r"\bmetric", r"\bbenchmark\b",
        r"\breason\b", r"\bcritiq", r"\bassess\b", r"\breview\b",
        r"\bweigh\b", r"\bvs\.?\b", r"\bversus\b",
        r"\bstrength", r"\bweakness", r"\bdifference\b",
    ],
    IntentCategory.CONVERSATIONAL: [
        r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bthanks\b",
        r"\bthank\s+you\b", r"\bgoodbye\b", r"\bbye\b",
        r"\bhow\s+are\s+you\b", r"\bchat\b", r"\btalk\b",
        r"\bsorry\b", r"\bplease\b", r"\bhelp\s+me\b",
        r"\bcan\s+you\b", r"\bcould\s+you\b", r"\bwould\s+you\b",
        r"\bjust\s+wondering\b", r"\bcurious\b",
    ],
}


# Intent descriptor sentences used for embedding strategy
_INTENT_DESCRIPTORS: Dict[IntentCategory, str] = {
    IntentCategory.INFORMATIONAL: (
        "Explain facts, provide definitions, give overviews and summaries of topics."
    ),
    IntentCategory.CREATIVE: (
        "Write stories, brainstorm ideas, create marketing copy, draft creative content."
    ),
    IntentCategory.TECHNICAL: (
        "Write code, debug errors, design software architecture, implement algorithms."
    ),
    IntentCategory.ANALYTICAL: (
        "Compare options, analyze data, evaluate trade-offs, assess pros and cons."
    ),
    IntentCategory.CONVERSATIONAL: (
        "Casual chat, greetings, small talk, polite conversation and interaction."
    ),
}


# ------------------------------------------------------------------
# Intent Engine
# ------------------------------------------------------------------

INTENT_STRATEGY: str = os.getenv("INTENT_STRATEGY", "rule").lower()


class IntentEngine:
    """Detects prompt intent and recommends an aggressiveness level.

    Usage::

        engine = IntentEngine()
        result = engine.detect(prompt_text)
        print(result.intent_label, result.recommended_aggressiveness)

    For the embedding strategy the engine needs a ``ModelLoader`` instance.
    Pass it via :pymethod:`detect` or :pymethod:`set_model_loader`.
    """

    def __init__(self) -> None:
        self._model_loader: Optional[Any] = None
        self._intent_embeddings: Optional[Dict[IntentCategory, Any]] = None

    # ------------------------------------------------------------------
    # Optional model loader for embedding strategy
    # ------------------------------------------------------------------

    def set_model_loader(self, model_loader: Any) -> None:
        """Attach a ``ModelLoader`` instance for embedding-based detection."""
        self._model_loader = model_loader

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        prompt: str,
        model_loader: Optional[Any] = None,
    ) -> IntentResult:
        """Classify *prompt* and return intent + recommended aggressiveness.

        Args:
            prompt: the raw user prompt text.
            model_loader: optional ``ModelLoader`` override for embedding mode.

        Returns:
            An ``IntentResult`` with label, confidence, and aggressiveness.
        """
        loader = model_loader or self._model_loader
        strategy = INTENT_STRATEGY

        if strategy == "embedding" and loader is not None:
            try:
                return self._detect_embedding(prompt, loader)
            except Exception:
                logger.warning(
                    "Embedding-based intent detection failed; falling back to rule-based.",
                    exc_info=True,
                )

        return self._detect_rule_based(prompt)

    # ------------------------------------------------------------------
    # Rule-based strategy
    # ------------------------------------------------------------------

    def _detect_rule_based(self, prompt: str) -> IntentResult:
        """Score each intent category via keyword hit-count."""
        text = prompt.lower()
        scores: Dict[IntentCategory, int] = {}

        for category, patterns in _KEYWORD_BANKS.items():
            hits = sum(1 for p in patterns if re.search(p, text))
            scores[category] = hits

        total_hits = sum(scores.values())

        if total_hits == 0:
            # No signal → default to INFORMATIONAL with low confidence
            best = IntentCategory.INFORMATIONAL
            confidence = 0.30
        else:
            best = max(scores, key=scores.get)  # type: ignore[arg-type]
            confidence = min(scores[best] / max(total_hits, 1), 1.0)
            # Scale confidence so a single hit isn't 100%
            confidence = min(0.40 + confidence * 0.60, 1.0)

        aggressiveness = self._pick_aggressiveness(best, confidence)

        logger.info(
            "Intent (rule-based): %s  confidence=%.2f  aggr=%.2f  hits=%s",
            best.value,
            confidence,
            aggressiveness,
            dict(scores),
        )

        return IntentResult(
            intent_label=best.value,
            confidence_score=round(confidence, 4),
            recommended_aggressiveness=round(aggressiveness, 4),
            strategy="rule",
        )

    # ------------------------------------------------------------------
    # Embedding-based strategy
    # ------------------------------------------------------------------

    def _detect_embedding(self, prompt: str, model_loader: Any) -> IntentResult:
        """Score each intent by cosine similarity to descriptor embeddings."""
        from app.utils.similarity import cosine_similarity_score, simple_sentence_embedding

        prompt_emb = simple_sentence_embedding(prompt, model_loader)

        # Lazily compute and cache descriptor embeddings
        if self._intent_embeddings is None:
            self._intent_embeddings = {}
            for cat, desc in _INTENT_DESCRIPTORS.items():
                self._intent_embeddings[cat] = simple_sentence_embedding(desc, model_loader)

        scores: Dict[IntentCategory, float] = {}
        for cat, cat_emb in self._intent_embeddings.items():
            scores[cat] = cosine_similarity_score(prompt_emb, cat_emb)

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best]

        # Normalise confidence to [0, 1] range
        # Cosine similarities between short texts tend to cluster around 0.7-0.95
        # so we rescale the gap between best and second-best
        sorted_scores = sorted(scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        confidence = min(0.50 + margin * 5.0, 1.0)  # amplify small margins

        aggressiveness = self._pick_aggressiveness(best, confidence)

        logger.info(
            "Intent (embedding): %s  sim=%.4f  confidence=%.2f  aggr=%.2f",
            best.value,
            best_score,
            confidence,
            aggressiveness,
        )

        return IntentResult(
            intent_label=best.value,
            confidence_score=round(confidence, 4),
            recommended_aggressiveness=round(aggressiveness, 4),
            strategy="embedding",
        )

    # ------------------------------------------------------------------
    # Aggressiveness selection
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_aggressiveness(intent: IntentCategory, confidence: float) -> float:
        """Choose aggressiveness within the intent's allowed range.

        Higher confidence → closer to the range default.
        Lower confidence  → shift toward conservative (lower) end.
        """
        lo, default, hi = _AGGRESSIVENESS_MAP[intent]

        if confidence >= 0.7:
            return default
        # Interpolate between lo and default based on confidence
        t = confidence / 0.7  # 0..1
        return round(lo + t * (default - lo), 4)
