"""Candidate generator — produces multiple optimised-prompt variants.

Instead of a single optimised output, the generator creates several
candidates using different strategies:

  1. **aggressive**  — high-aggressiveness token compression.
  2. **balanced**    — moderate token compression + phrase densification.
  3. **semantic**    — phrase densification only (no token pruning).
  4. **structured**  — balanced + prompt restructuring.

Downstream, the decision engine picks the best candidate based on
a composite score of similarity, compression ratio, and density.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.compressor import Compressor, CompressionResult
from app.core.densifier import Densifier, DensificationResult
from app.core.prompt_structurer import PromptStructurer, StructuringResult
from app.core.model_loader import ModelLoader

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class Candidate:
    """A single prompt candidate with its generation metadata."""

    strategy: str
    text: str
    token_compression: Optional[CompressionResult] = None
    densification: Optional[DensificationResult] = None
    structuring: Optional[StructuringResult] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "strategy": self.strategy,
            "text": self.text,
        }
        if self.token_compression is not None:
            d["token_reduction_percent"] = round(
                self.token_compression.token_reduction_percent, 4
            )
        if self.densification is not None:
            d["phrase_rules_applied"] = self.densification.phrase_reduction_count
        if self.structuring is not None:
            d["was_restructured"] = self.structuring.was_restructured
        return d


@dataclass
class CandidateSet:
    """Full set of candidates generated for a prompt."""

    original_text: str
    candidates: List[Candidate] = field(default_factory=list)

    def texts(self) -> List[str]:
        return [c.text for c in self.candidates]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": len(self.candidates),
            "candidates": [c.to_dict() for c in self.candidates],
        }


# ------------------------------------------------------------------
# Strategy aggressiveness map
# ------------------------------------------------------------------

_STRATEGY_AGGR: Dict[str, float] = {
    "aggressive": 0.85,
    "balanced": 0.45,
    "semantic": 0.0,    # no token pruning
    "structured": 0.45,
}


# ------------------------------------------------------------------
# Generator
# ------------------------------------------------------------------

class CandidateGenerator:
    """Generates multiple optimised prompt candidates."""

    def __init__(self, model_loader: ModelLoader) -> None:
        self._model = model_loader
        self._compressor = Compressor(model_loader)
        self._densifier = Densifier()
        self._structurer = PromptStructurer()

    def generate(
        self,
        original_text: str,
        intent_label: Optional[str] = None,
        base_aggressiveness: float = 0.5,
        mode: str = "optimize",
        prefer_structuring: bool = False,
        skip_aggressive: bool = False,
    ) -> CandidateSet:
        """Produce a ``CandidateSet`` for *original_text*.

        The *base_aggressiveness* (from intent detection) is used to
        scale each strategy's default aggressiveness so the overall
        tone matches the intent.

        Args:
            original_text: the raw user prompt.
            intent_label: detected intent category (e.g. ``"TECHNICAL"``).
            base_aggressiveness: recommended aggressiveness from the intent
                engine.  Strategies are scaled relative to this value.

        Returns:
            A ``CandidateSet`` containing one ``Candidate`` per strategy.
        """
        candidates: List[Candidate] = []
        strategy_names = self._strategies_for_mode(
            mode=mode,
            prefer_structuring=prefer_structuring,
            skip_aggressive=skip_aggressive,
        )

        for strategy in strategy_names:
            try:
                candidate = self._build_candidate(
                    strategy=strategy,
                    text=original_text,
                    intent_label=intent_label,
                    base_aggressiveness=base_aggressiveness,
                    default_aggr=_STRATEGY_AGGR[strategy],
                    mode=mode,
                )
                candidates.append(candidate)
            except Exception:
                logger.warning(
                    "Candidate strategy '%s' failed — skipping.",
                    strategy,
                    exc_info=True,
                )

        cset = CandidateSet(original_text=original_text, candidates=candidates)
        logger.info(
            "Generated %d candidates for prompt of %d chars",
            len(candidates),
            len(original_text),
        )
        return cset

    @staticmethod
    def _strategies_for_mode(
        mode: str,
        prefer_structuring: bool,
        skip_aggressive: bool,
    ) -> List[str]:
        """Return the ordered strategy set for the requested optimisation mode."""
        normalized = (mode or "optimize").lower()

        if normalized == "enhance":
            strategies = ["semantic", "balanced"]
        elif normalized == "both":
            strategies = ["balanced", "semantic"]
            if not skip_aggressive:
                strategies.insert(0, "aggressive")
        else:
            strategies = ["balanced", "semantic"]
            if not skip_aggressive:
                strategies.insert(0, "aggressive")

        if prefer_structuring and "structured" not in strategies:
            strategies.append("structured")

        if normalized == "enhance":
            strategies = [s for s in strategies if s != "aggressive"]

        # De-duplicate while preserving order.
        deduped: List[str] = []
        for name in strategies:
            if name not in deduped:
                deduped.append(name)
        return deduped

    # ------------------------------------------------------------------
    # Private: build a single candidate
    # ------------------------------------------------------------------

    def _build_candidate(
        self,
        strategy: str,
        text: str,
        intent_label: Optional[str],
        base_aggressiveness: float,
        default_aggr: float,
        mode: str,
    ) -> Candidate:
        """Build one ``Candidate`` according to *strategy*."""

        # Scale the strategy's default aggr by the intent-recommended base.
        # e.g. if base=0.2 (TECHNICAL) and default=0.85 (aggressive),
        # scaled → 0.2 + (0.85 - 0.45)*(0.2/0.5) ≈ 0.36  — much tamer.
        scale = base_aggressiveness / 0.5 if base_aggressiveness > 0 else 1.0
        aggr = max(0.0, min(1.0, default_aggr * scale))

        compression: Optional[CompressionResult] = None
        densification: Optional[DensificationResult] = None
        structuring: Optional[StructuringResult] = None
        current = text

        # ---- Token-level compression (skip for 'semantic' strategy) ----
        normalized_mode = (mode or "optimize").lower()

        if strategy != "semantic" and normalized_mode != "enhance":
            compression = self._compressor.compress_prompt(current, aggressiveness=aggr)
            current = compression.compressed_text

        # ---- Phrase-level densification (all strategies) ----
        densification = self._densifier.densify(
            current,
            intent_label=intent_label,
            aggressiveness=aggr,
        )
        current = densification.densified_text

        # ---- Structuring (only for 'structured' strategy) ----
        if strategy == "structured":
            structuring = self._structurer.structure(current, intent_label=intent_label)
            current = structuring.structured_text

        return Candidate(
            strategy=strategy,
            text=current,
            token_compression=compression,
            densification=densification,
            structuring=structuring,
        )
