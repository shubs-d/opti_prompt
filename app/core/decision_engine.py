"""Decision engine — approves or rejects compressions and selects the best candidate.

Extends the original rules-based gate with a composite scoring system that
evaluates multiple candidates using weighted similarity, compression ratio,
and information density — informed by intent-specific strategy weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Decision(str, Enum):
    """Possible compression decisions."""

    APPROVE = "APPROVE"
    REJECT = "REJECT"
    CONSERVATIVE_REQUIRED = "CONSERVATIVE_REQUIRED"
    FALLBACK = "FALLBACK"


# ------------------------------------------------------------------
# Default thresholds (can be overridden via constructor)
# ------------------------------------------------------------------
DEFAULT_MAX_DRIFT: float = 0.15
DEFAULT_CONSERVATIVE_DRIFT: float = 0.08
DEFAULT_MIN_REDUCTION: float = 5.0
DEFAULT_MAX_REDUCTION: float = 70.0


# ------------------------------------------------------------------
# Candidate selection data
# ------------------------------------------------------------------

@dataclass
class CandidateScore:
    """Scored candidate ready for ranking."""

    strategy: str
    text: str
    similarity: float
    compression_ratio: float
    density_score: float
    composite_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "similarity": round(self.similarity, 6),
            "compression_ratio": round(self.compression_ratio, 4),
            "density_score": round(self.density_score, 6),
            "composite_score": round(self.composite_score, 6),
        }


class DecisionEngine:
    """Rules-based gate + multi-candidate ranking.

    Single-candidate mode (legacy ``decide`` method):
        * ``max_drift``: drift above this → REJECT.
        * ``conservative_drift``: drift above this → CONSERVATIVE_REQUIRED.
        * ``min_reduction``: reduction below this → REJECT (not worth it).
        * ``max_reduction``: reduction above this → REJECT (too aggressive).

    Multi-candidate mode (``select_best``):
        Ranks candidates by a weighted composite of similarity,
        compression ratio, and density, then applies the quality gate
        on the winner.
    """

    def __init__(
        self,
        max_drift: float = DEFAULT_MAX_DRIFT,
        conservative_drift: float = DEFAULT_CONSERVATIVE_DRIFT,
        min_reduction: float = DEFAULT_MIN_REDUCTION,
        max_reduction: float = DEFAULT_MAX_REDUCTION,
    ) -> None:
        self.max_drift = max_drift
        self.conservative_drift = conservative_drift
        self.min_reduction = min_reduction
        self.max_reduction = max_reduction

    # ==================================================================
    # Legacy single-candidate gate
    # ==================================================================

    def decide(
        self,
        token_reduction_percent: float,
        drift_score: float,
    ) -> Dict[str, Any]:
        """Evaluate compression metrics and return a decision.

        Args:
            token_reduction_percent: percentage of tokens removed.
            drift_score: semantic drift in [0, 1].

        Returns:
            Dict with ``decision``, ``reason``, and input metrics.
        """
        decision: Decision
        reason: str

        if drift_score > self.max_drift:
            decision = Decision.REJECT
            reason = (
                f"Drift score {drift_score:.4f} exceeds maximum "
                f"allowed drift {self.max_drift}."
            )
        elif token_reduction_percent > self.max_reduction:
            decision = Decision.REJECT
            reason = (
                f"Token reduction {token_reduction_percent:.1f}% exceeds "
                f"safety ceiling of {self.max_reduction}%."
            )
        elif token_reduction_percent < self.min_reduction:
            decision = Decision.REJECT
            reason = (
                f"Token reduction {token_reduction_percent:.1f}% is below "
                f"minimum useful threshold of {self.min_reduction}%."
            )
        elif drift_score > self.conservative_drift:
            decision = Decision.CONSERVATIVE_REQUIRED
            reason = (
                f"Drift score {drift_score:.4f} is acceptable but above "
                f"conservative threshold {self.conservative_drift}. "
                "Consider reducing aggressiveness."
            )
        else:
            decision = Decision.APPROVE
            reason = (
                f"Compression looks good — {token_reduction_percent:.1f}% reduction "
                f"with only {drift_score:.4f} drift."
            )

        logger.info("Decision: %s — %s", decision.value, reason)

        return {
            "decision": decision.value,
            "reason": reason,
            "token_reduction_percent": round(token_reduction_percent, 4),
            "drift_score": round(drift_score, 6),
        }

    # ==================================================================
    # Multi-candidate selection
    # ==================================================================

    def select_best(
        self,
        candidates: List[Dict[str, Any]],
        original_text: str,
        min_similarity: float = 0.85,
        similarity_weight: float = 0.40,
        compression_weight: float = 0.20,
        density_weight: float = 0.40,
    ) -> Dict[str, Any]:
        """Rank *candidates* and select the best one.

        Each entry in *candidates* must contain:
            * ``strategy`` (str)
            * ``text`` (str)
            * ``similarity`` (float)  — semantic similarity to original
            * ``compression_ratio`` (float) — compressed / original token ratio
            * ``density_score`` (float)

        Args:
            candidates: scored candidate dicts.
            original_text: the raw user prompt (used as fallback).
            min_similarity: reject candidates below this similarity.
            similarity_weight: composite weight for similarity.
            compression_weight: composite weight for compression savings.
            density_weight: composite weight for density.

        Returns:
            Dict with ``decision``, ``selected_strategy``, ``selected_text``,
            ``composite_score``, ``reason``, and ``all_scores``.
        """
        scored: List[CandidateScore] = []
        for c in candidates:
            sim = c["similarity"]
            comp_ratio = c["compression_ratio"]
            density = c["density_score"]

            # Skip candidates that fail the similarity floor
            if sim < min_similarity:
                logger.info(
                    "Candidate '%s' below similarity floor (%.4f < %.4f)",
                    c["strategy"],
                    sim,
                    min_similarity,
                )
                continue

            # Composite: higher is better.
            # compression_savings = 1 - ratio (more savings = better)
            compression_savings = max(1.0 - comp_ratio, 0.0)
            composite = (
                similarity_weight * sim
                + compression_weight * compression_savings
                + density_weight * density
            )

            scored.append(CandidateScore(
                strategy=c["strategy"],
                text=c["text"],
                similarity=sim,
                compression_ratio=comp_ratio,
                density_score=density,
                composite_score=composite,
            ))

        if not scored:
            # Every candidate failed — fall back to original
            logger.warning("All candidates below similarity floor; falling back to original.")
            return {
                "decision": Decision.FALLBACK.value,
                "selected_strategy": "original",
                "selected_text": original_text,
                "composite_score": 0.0,
                "reason": (
                    f"All {len(candidates)} candidates fell below the similarity "
                    f"floor of {min_similarity:.2f}. Returning original prompt."
                ),
                "all_scores": [],
            }

        # Rank by composite (descending)
        scored.sort(key=lambda s: s.composite_score, reverse=True)
        best = scored[0]

        # Apply the legacy quality gate on the winner's metrics
        drift = 1.0 - best.similarity
        reduction_pct = (1.0 - best.compression_ratio) * 100.0
        gate = self.decide(
            token_reduction_percent=reduction_pct,
            drift_score=drift,
        )

        # If the best candidate is REJECTED, fall back to the original
        if gate["decision"] == Decision.REJECT.value:
            # Try the next best candidate that passes
            for runner_up in scored[1:]:
                ru_drift = 1.0 - runner_up.similarity
                ru_red = (1.0 - runner_up.compression_ratio) * 100.0
                ru_gate = self.decide(token_reduction_percent=ru_red, drift_score=ru_drift)
                if ru_gate["decision"] != Decision.REJECT.value:
                    best = runner_up
                    gate = ru_gate
                    break
            else:
                # All rejected → fallback
                return {
                    "decision": Decision.FALLBACK.value,
                    "selected_strategy": "original",
                    "selected_text": original_text,
                    "composite_score": 0.0,
                    "reason": "All candidates rejected by quality gate. Returning original.",
                    "all_scores": [s.to_dict() for s in scored],
                }

        logger.info(
            "Selected candidate '%s' — composite=%.4f  sim=%.4f  density=%.4f",
            best.strategy,
            best.composite_score,
            best.similarity,
            best.density_score,
        )

        return {
            "decision": gate["decision"],
            "selected_strategy": best.strategy,
            "selected_text": best.text,
            "composite_score": round(best.composite_score, 6),
            "reason": gate["reason"],
            "all_scores": [s.to_dict() for s in scored],
        }

