"""Lightweight reflection logic that adapts GEPA exploration pressure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.core.gepa.population import PromptCandidate


@dataclass(slots=True)
class ReflectionState:
    """Tracks progress trends across generations."""

    best_drift: float
    best_reduction: float
    stagnant_generations: int = 0


class ReflectionController:
    """Monitors generation improvements and adjusts mutation focus."""

    def __init__(self) -> None:
        self.state: ReflectionState | None = None

    def update(self, best_candidate: PromptCandidate) -> ReflectionState:
        """Update reflection state and return new controller state."""
        metrics = best_candidate.metrics
        if metrics is None:
            if self.state is None:
                self.state = ReflectionState(best_drift=1.0, best_reduction=0.0)
            return self.state

        if self.state is None:
            self.state = ReflectionState(
                best_drift=metrics.drift_score,
                best_reduction=metrics.token_reduction_percent,
                stagnant_generations=0,
            )
            return self.state

        improved = (
            metrics.drift_score < self.state.best_drift - 0.002
            or metrics.token_reduction_percent > self.state.best_reduction + 0.5
        )
        if improved:
            self.state.best_drift = min(self.state.best_drift, metrics.drift_score)
            self.state.best_reduction = max(self.state.best_reduction, metrics.token_reduction_percent)
            self.state.stagnant_generations = 0
        else:
            self.state.stagnant_generations += 1

        return self.state

    def parent_pool_size(self, default_size: int) -> int:
        """Increase exploration width when optimization stagnates."""
        if self.state is None:
            return default_size
        if self.state.stagnant_generations >= 2:
            return min(default_size + 2, 8)
        return default_size

    def trim_limit(self, population_size: int) -> int:
        """Keep more elites when progression slows."""
        if self.state is None:
            return max(2, population_size // 2)
        if self.state.stagnant_generations >= 2:
            return max(3, (population_size // 2) + 1)
        return max(2, population_size // 2)


def summarize_generation(generation: int, candidate: PromptCandidate, frontier_size: int) -> str:
    """Create compact debug summary for logs and API tracing."""
    if candidate.metrics is None:
        return f"gen={generation} no-metrics frontier={frontier_size}"

    return (
        f"gen={generation} frontier={frontier_size} "
        f"drift={candidate.metrics.drift_score:.4f} "
        f"sim={candidate.metrics.semantic_similarity:.4f} "
        f"reduction={candidate.metrics.token_reduction_percent:.1f}%"
    )


def sort_by_quality(candidates: List[PromptCandidate]) -> List[PromptCandidate]:
    """Sort candidates by quality objectives for deterministic slicing."""
    return sorted(
        [candidate for candidate in candidates if candidate.metrics is not None],
        key=lambda item: (
            item.metrics.drift_score,
            -item.metrics.semantic_similarity,
            -item.metrics.token_reduction_percent,
        ),
    )
