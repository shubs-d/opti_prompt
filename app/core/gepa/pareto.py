"""Pareto-frontier utilities for GEPA candidate selection."""

from __future__ import annotations

from typing import List

from app.core.gepa.population import PromptCandidate


def dominates(left: PromptCandidate, right: PromptCandidate) -> bool:
    """Return True when `left` dominates `right` in objective space."""
    if left.metrics is None or right.metrics is None:
        return False

    no_worse = (
        left.metrics.token_reduction_percent >= right.metrics.token_reduction_percent
        and left.metrics.drift_score <= right.metrics.drift_score
    )
    strictly_better = (
        left.metrics.token_reduction_percent > right.metrics.token_reduction_percent
        or left.metrics.drift_score < right.metrics.drift_score
    )
    return no_worse and strictly_better


def pareto_frontier(candidates: List[PromptCandidate]) -> List[PromptCandidate]:
    """Return all non-dominated candidates."""
    frontier: List[PromptCandidate] = []

    for candidate in candidates:
        if candidate.metrics is None:
            continue
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            if dominates(other, candidate):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)

    return frontier


def utility_score(candidate: PromptCandidate) -> float:
    """Secondary ranking score for deterministic tie-breaking on frontier."""
    if candidate.metrics is None:
        return float("-inf")

    return (
        0.50 * candidate.metrics.token_reduction_percent
        + 35.0 * candidate.metrics.semantic_similarity
        - 40.0 * candidate.metrics.drift_score
    )


def pick_best(frontier: List[PromptCandidate], fallback: PromptCandidate) -> PromptCandidate:
    """Pick best candidate from the frontier, favoring safe drift."""
    if not frontier:
        return fallback

    safe = [candidate for candidate in frontier if candidate.metrics and candidate.metrics.drift_score <= 0.15]
    pool = safe or frontier
    return max(pool, key=utility_score)
