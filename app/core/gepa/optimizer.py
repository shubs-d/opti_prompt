"""GEPA optimizer orchestration.

Pipeline:
1. Seed population from baseline compressed prompt.
2. Mutate candidates across generations.
3. Evaluate with existing Evaluator metrics.
4. Keep Pareto frontier (maximize reduction, minimize drift).
5. Return best prompt under a small runtime budget.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List

from app.core.compressor import Compressor
from app.core.evaluator import EvaluationReport, Evaluator
from app.core.gepa.mutation import MutationEngine
from app.core.gepa.pareto import pareto_frontier, pick_best
from app.core.gepa.population import CandidateMetrics, Population, PromptCandidate
from app.core.gepa.reflection import ReflectionController, sort_by_quality, summarize_generation
from app.core.model_loader import ModelLoader

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GepaOptimizationResult:
    """Final GEPA output consumed by API pipeline."""

    best_prompt: str
    best_report: EvaluationReport
    generations_run: int
    frontier_size: int
    baseline_prompt: str
    baseline_report: EvaluationReport
    summaries: List[str]


class GepaOptimizer:
    """Genetic / evolutionary prompt optimization layer."""

    def __init__(
        self,
        model_loader: ModelLoader,
        compressor: Compressor,
        evaluator: Evaluator,
    ) -> None:
        self._model_loader = model_loader
        self._compressor = compressor
        self._evaluator = evaluator
        self._mutation = MutationEngine(compressor=compressor)
        self._reflection = ReflectionController()

    def optimize(
        self,
        original_prompt: str,
        baseline_prompt: str,
        test_query: str,
        base_aggressiveness: float,
        generations: int = 6,
        population_size: int = 6,
        time_budget_seconds: float = 1.5,
    ) -> GepaOptimizationResult:
        """Run a bounded GEPA search and return the selected prompt."""
        generations = max(1, min(generations, 10))
        population_size = max(4, min(population_size, 12))
        time_budget_seconds = max(0.4, min(time_budget_seconds, 3.0))

        start = time.perf_counter()

        baseline_report = self._evaluator.evaluate(
            original_prompt=original_prompt,
            compressed_prompt=baseline_prompt,
            test_query=test_query,
        )

        population = Population(candidates=self._seed_population(
            original_prompt=original_prompt,
            baseline_prompt=baseline_prompt,
            base_aggressiveness=base_aggressiveness,
            population_size=population_size,
        ))

        self._evaluate_population(population, original_prompt, test_query)

        evaluated = [candidate for candidate in population.candidates if candidate.metrics is not None]
        if not evaluated:
            return GepaOptimizationResult(
                best_prompt=baseline_prompt,
                best_report=baseline_report,
                generations_run=0,
                frontier_size=0,
                baseline_prompt=baseline_prompt,
                baseline_report=baseline_report,
                summaries=["GEPA: no candidates evaluated; baseline returned."],
            )

        best_candidate = min(
            evaluated,
            key=lambda item: (item.metrics.drift_score, -item.metrics.token_reduction_percent),
        )
        summaries: List[str] = []
        frontier = pareto_frontier(evaluated)

        for generation in range(1, generations + 1):
            elapsed = time.perf_counter() - start
            if elapsed >= time_budget_seconds:
                logger.info("GEPA stopped early due to time budget (%.3fs)", elapsed)
                break

            state = self._reflection.update(best_candidate)
            parent_limit = self._reflection.parent_pool_size(default_size=max(2, population_size // 2))
            parents = sort_by_quality(frontier)[:parent_limit]
            if not parents:
                break

            offspring: List[PromptCandidate] = []
            for parent in parents:
                offspring.extend(
                    self._mutation.mutate(
                        original_prompt=original_prompt,
                        parent=parent,
                        next_generation=generation,
                    )
                )

            for child in offspring:
                population.add(child)

            self._evaluate_population(population, original_prompt, test_query)

            evaluated = [candidate for candidate in population.candidates if candidate.metrics is not None]
            frontier = pareto_frontier(evaluated)
            best_candidate = pick_best(frontier, fallback=best_candidate)

            summaries.append(summarize_generation(generation, best_candidate, len(frontier)))

            trim_to = self._reflection.trim_limit(population_size)
            retained = sort_by_quality(frontier)[:trim_to]
            if best_candidate not in retained:
                retained.append(best_candidate)
            population = Population(candidates=retained)

            if state.stagnant_generations >= 3 and generation >= 4:
                logger.info("GEPA convergence detected at generation %d", generation)
                break

        best_report = self._evaluator.evaluate(
            original_prompt=original_prompt,
            compressed_prompt=best_candidate.prompt,
            test_query=test_query,
        )

        # Never return a candidate that degrades drift significantly vs baseline.
        if best_report.drift_score > baseline_report.drift_score + 0.01:
            return GepaOptimizationResult(
                best_prompt=baseline_prompt,
                best_report=baseline_report,
                generations_run=len(summaries),
                frontier_size=len(frontier),
                baseline_prompt=baseline_prompt,
                baseline_report=baseline_report,
                summaries=summaries + ["GEPA fallback to baseline due to drift regression."],
            )

        return GepaOptimizationResult(
            best_prompt=best_candidate.prompt,
            best_report=best_report,
            generations_run=len(summaries),
            frontier_size=len(frontier),
            baseline_prompt=baseline_prompt,
            baseline_report=baseline_report,
            summaries=summaries,
        )

    def _seed_population(
        self,
        original_prompt: str,
        baseline_prompt: str,
        base_aggressiveness: float,
        population_size: int,
    ) -> List[PromptCandidate]:
        seeds: List[PromptCandidate] = [
            PromptCandidate(
                prompt=baseline_prompt,
                aggressiveness=base_aggressiveness,
                threshold_scale=1.0,
                origin="baseline",
                generation=0,
            )
        ]

        for aggr in [max(0.05, base_aggressiveness - 0.15), base_aggressiveness, min(0.95, base_aggressiveness + 0.15)]:
            compressed = self._compressor.compress_prompt(original_prompt, aggressiveness=aggr)
            seeds.append(
                PromptCandidate(
                    prompt=compressed.compressed_text,
                    aggressiveness=aggr,
                    threshold_scale=1.0,
                    origin="seed_recompress",
                    generation=0,
                )
            )

        # Deduplicate while preserving order.
        unique: List[PromptCandidate] = []
        seen = set()
        for seed in seeds:
            if seed.prompt not in seen:
                seen.add(seed.prompt)
                unique.append(seed)
            if len(unique) >= population_size:
                break

        return unique

    def _evaluate_population(self, population: Population, original_prompt: str, test_query: str) -> None:
        _, original_tokens = self._model_loader.tokenize(original_prompt)
        original_count = max(len(original_tokens), 1)

        for candidate in population.candidates:
            if candidate.metrics is not None:
                continue

            report = self._evaluator.evaluate(
                original_prompt=original_prompt,
                compressed_prompt=candidate.prompt,
                test_query=test_query,
            )
            _, candidate_tokens = self._model_loader.tokenize(candidate.prompt)
            reduction = (1.0 - (len(candidate_tokens) / original_count)) * 100.0
            reduction = max(0.0, reduction)

            candidate.metrics = CandidateMetrics(
                drift_score=report.drift_score,
                semantic_similarity=report.semantic_similarity,
                token_reduction_percent=reduction,
            )
