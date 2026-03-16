"""Population data model for GEPA prompt evolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(slots=True)
class CandidateMetrics:
    """Evaluation metrics tracked for each evolved prompt candidate."""

    drift_score: float
    semantic_similarity: float
    token_reduction_percent: float


@dataclass(slots=True)
class PromptCandidate:
    """Single member of the GEPA population."""

    prompt: str
    aggressiveness: float
    threshold_scale: float
    origin: str
    generation: int
    metrics: CandidateMetrics | None = None


class Population:
    """Container for prompt candidates with deduplication helpers."""

    def __init__(self, candidates: List[PromptCandidate] | None = None) -> None:
        self.candidates: List[PromptCandidate] = candidates or []

    def add(self, candidate: PromptCandidate) -> bool:
        """Add candidate if prompt is not already present."""
        if any(existing.prompt == candidate.prompt for existing in self.candidates):
            return False
        self.candidates.append(candidate)
        return True

    def extend(self, candidates: List[PromptCandidate]) -> None:
        for candidate in candidates:
            self.add(candidate)

    def top(self, limit: int) -> List[PromptCandidate]:
        """Return top candidates sorted by low drift then high reduction."""
        scored = [candidate for candidate in self.candidates if candidate.metrics is not None]
        scored.sort(
            key=lambda item: (
                item.metrics.drift_score,
                -item.metrics.token_reduction_percent,
                -item.metrics.semantic_similarity,
            )
        )
        return scored[:limit]
