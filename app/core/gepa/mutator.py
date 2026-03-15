"""GEPA mutator — reflective candidate generation + Pareto selection.

This module treats reflection as a lightweight genetic step:
  1. Create multiple mutation hints (different repair strategies).
  2. Ask the reflection LLM for repaired candidates.
  3. Evaluate each candidate on drift, compression, and density.
  4. Keep Pareto-optimal candidates, then select a weighted best candidate.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.core.density_metrics import DensityMetrics
from app.core.evaluator import EvaluationReport, Evaluator
from app.core.gepa.reflection_llm import ReflectionLLM, ReflectionResult, build_reflection_llm_from_env
from app.core.model_loader import ModelLoader


@dataclass
class GepaCandidate:
    """Single GEPA repair candidate with evaluation metrics."""

    prompt: str
    reasoning: str
    provider: str
    model: str
    drift_score: float
    semantic_similarity: float
    compression_ratio: float
    density_score: float
    pareto_utility: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "reasoning": self.reasoning,
            "provider": self.provider,
            "model": self.model,
            "drift_score": round(self.drift_score, 6),
            "semantic_similarity": round(self.semantic_similarity, 6),
            "compression_ratio": round(self.compression_ratio, 4),
            "density_score": round(self.density_score, 6),
            "pareto_utility": round(self.pareto_utility, 6),
        }


@dataclass
class GepaRepairOutcome:
    """Outcome of a GEPA repair attempt."""

    applied: bool
    repaired_prompt: str
    final_report: EvaluationReport
    reflection_reasoning: str
    candidates: List[GepaCandidate]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "applied": self.applied,
            "repaired_prompt": self.repaired_prompt,
            "final_drift_score": round(self.final_report.drift_score, 6),
            "final_similarity": round(self.final_report.semantic_similarity, 6),
            "reflection_reasoning": self.reflection_reasoning,
            "candidates": [c.to_dict() for c in self.candidates],
        }


class GepaMutator:
    """Orchestrates GEPA reflective repair for drifted compressed prompts."""

    def __init__(
        self,
        model_loader: ModelLoader,
        evaluator: Optional[Evaluator] = None,
        reflection_llm: Optional[ReflectionLLM] = None,
    ) -> None:
        self._model_loader = model_loader
        self._evaluator = evaluator or Evaluator(model_loader)
        self._density = DensityMetrics(model_loader)
        self._reflection_llm = reflection_llm or build_reflection_llm_from_env()

    async def repair_prompt(
        self,
        original_prompt: str,
        broken_prompt: str,
        test_query: str,
        max_candidates: int = 3,
        token_budget_ratio: float = 0.72,
    ) -> GepaRepairOutcome:
        """Generate and select repaired prompts using GEPA-style mutation."""
        max_candidates = max(1, min(max_candidates, 6))
        token_budget_ratio = max(0.4, min(token_budget_ratio, 1.0))

        _, orig_tokens = self._model_loader.tokenize(original_prompt)
        target_tokens = max(1, int(len(orig_tokens) * token_budget_ratio))

        hints = self._build_mutation_hints(max_candidates)
        tasks = [
            self._reflection_llm.reflect(
                original_prompt=original_prompt,
                broken_prompt=broken_prompt,
                mutation_hint=hint,
                target_token_count=target_tokens,
            )
            for hint in hints
        ]

        reflections = await asyncio.gather(*tasks, return_exceptions=True)

        candidates: List[GepaCandidate] = []
        for item in reflections:
            if isinstance(item, Exception):
                continue
            candidate = self._score_candidate(
                reflection=item,
                original_prompt=original_prompt,
                test_query=test_query,
            )
            candidates.append(candidate)

        # Fallback candidate keeps the broken prompt if every reflection call failed.
        if not candidates:
            fallback_report = self._evaluator.evaluate(
                original_prompt=original_prompt,
                compressed_prompt=broken_prompt,
                test_query=test_query,
            )
            return GepaRepairOutcome(
                applied=False,
                repaired_prompt=broken_prompt,
                final_report=fallback_report,
                reflection_reasoning="GEPA reflection failed; returned broken compressed prompt.",
                candidates=[],
            )

        selected = self._select_best_candidate(candidates)
        final_report = self._evaluator.evaluate(
            original_prompt=original_prompt,
            compressed_prompt=selected.prompt,
            test_query=test_query,
        )

        return GepaRepairOutcome(
            applied=True,
            repaired_prompt=selected.prompt,
            final_report=final_report,
            reflection_reasoning=selected.reasoning,
            candidates=candidates,
        )

    @staticmethod
    def _build_mutation_hints(max_candidates: int) -> List[str]:
        pool = [
            "Reinsert missing constraints and success criteria, then compact wording.",
            "Prioritize lost domain-specific details and remove redundant politeness.",
            "Restore missing structural instructions (format, steps, edge cases).",
            "Repair dropped entities, numbers, and required output style while staying concise.",
            "Recover hidden assumptions and validation conditions with minimal verbosity.",
            "Preserve intent-critical nouns/verbs first, then compress around them.",
        ]
        return pool[:max_candidates]

    def _score_candidate(
        self,
        reflection: ReflectionResult,
        original_prompt: str,
        test_query: str,
    ) -> GepaCandidate:
        prompt = reflection.repaired_prompt.strip()
        if not prompt:
            prompt = original_prompt

        eval_report = self._evaluator.evaluate(
            original_prompt=original_prompt,
            compressed_prompt=prompt,
            test_query=test_query,
        )
        density = self._density.score(original_prompt=original_prompt, candidate_text=prompt)

        # Utility over Pareto front for deterministic tie-breaking.
        similarity_component = eval_report.semantic_similarity
        compression_component = max(1.0 - density.compression_ratio, 0.0)
        density_component = density.density_score
        drift_penalty = eval_report.drift_score

        utility = (
            0.45 * similarity_component
            + 0.20 * compression_component
            + 0.25 * density_component
            - 0.30 * drift_penalty
        )

        return GepaCandidate(
            prompt=prompt,
            reasoning=reflection.reasoning,
            provider=reflection.provider,
            model=reflection.model,
            drift_score=eval_report.drift_score,
            semantic_similarity=eval_report.semantic_similarity,
            compression_ratio=density.compression_ratio,
            density_score=density.density_score,
            pareto_utility=utility,
        )

    def _select_best_candidate(self, candidates: List[GepaCandidate]) -> GepaCandidate:
        frontier = self._pareto_frontier(candidates)

        # Prefer frontier candidates that satisfy the safety drift ceiling.
        safe_frontier = [c for c in frontier if c.drift_score <= 0.15]
        selection_pool = safe_frontier or frontier

        selection_pool.sort(key=lambda c: (c.pareto_utility, -c.drift_score), reverse=True)
        return selection_pool[0]

    @staticmethod
    def _pareto_frontier(candidates: List[GepaCandidate]) -> List[GepaCandidate]:
        """Return non-dominated candidates.

        Objectives:
          * minimize drift_score
          * minimize compression_ratio
          * maximize density_score
        """
        frontier: List[GepaCandidate] = []

        for candidate in candidates:
            dominated = False
            for other in candidates:
                if other is candidate:
                    continue

                no_worse = (
                    other.drift_score <= candidate.drift_score
                    and other.compression_ratio <= candidate.compression_ratio
                    and other.density_score >= candidate.density_score
                )
                strictly_better = (
                    other.drift_score < candidate.drift_score
                    or other.compression_ratio < candidate.compression_ratio
                    or other.density_score > candidate.density_score
                )

                if no_worse and strictly_better:
                    dominated = True
                    break

            if not dominated:
                frontier.append(candidate)

        return frontier
