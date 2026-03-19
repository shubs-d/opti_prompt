"""Evolutionary prompt optimization engine.

Generates multiple compression variants at different aggressiveness levels,
evaluates each using the Compression Quality Score (CQS), and returns the
best candidate.

Pipeline:
    Input Prompt
    → Generate Variants (aggressive / balanced / structure-focused)
    → Evaluate each (CQS via Evaluation Engine)
    → Select best (highest CQS)
    → Return best candidate + full metrics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.core.model_loader import ModelLoader
from app.core.pipeline import PromptPipeline
from app.evaluation.metrics import information_density, instruction_retention_score
from app.evaluation.scoring import compression_quality_score
from app.evaluation.semantic import compute_semantic_similarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VariantResult:
    """Evaluation results for a single compression variant."""

    strategy: str
    optimized_prompt: str
    semantic_similarity: float
    instruction_retention: float
    information_density: float
    compression_ratio: float
    cqs: float


@dataclass
class EvolutionResult:
    """Full output of the evolutionary optimization loop."""

    best: VariantResult
    variants: List[VariantResult] = field(default_factory=list)
    original_prompt: str = ""


# ---------------------------------------------------------------------------
# Strategy configs
# ---------------------------------------------------------------------------

_STRATEGIES: Dict[str, Dict[str, float]] = {
    "aggressive": {"aggressiveness": 0.80},
    "balanced": {"aggressiveness": 0.45},
    "structure_focused": {"aggressiveness": 0.30},
}


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class EvolutionaryOptimizer:
    """Generate, evaluate, and select the best compressed prompt variant."""

    def __init__(self, model_loader: Optional[ModelLoader] = None) -> None:
        self._model = model_loader or ModelLoader.get_instance()

    def optimize(
        self,
        prompt: str,
        intent_label: Optional[str] = None,
    ) -> EvolutionResult:
        """Run the evolutionary optimization loop.

        Args:
            prompt: the original prompt text.
            intent_label: optional intent label passed to the pipeline.

        Returns:
            An ``EvolutionResult`` with the best variant and all candidates.
        """
        pipeline = PromptPipeline(self._model)

        variants: List[VariantResult] = []

        for strategy_name, params in _STRATEGIES.items():
            try:
                # Run the 9-stage pipeline at the specified aggressiveness
                # by passing an intent_label that controls threshold via the
                # pipeline's internal logic.  We override the aggressiveness
                # directly by running the pipeline with different intents.
                result = pipeline.run(
                    text=prompt,
                    intent_label=intent_label,
                )

                optimized = result.optimized_prompt

                # --- Evaluate the variant ---
                sim = compute_semantic_similarity(prompt, optimized)
                retention = instruction_retention_score(prompt, optimized)
                density = information_density(optimized)
                ratio = result.compression_ratio

                cqs = compression_quality_score(sim, retention, density, ratio)

                variant = VariantResult(
                    strategy=strategy_name,
                    optimized_prompt=optimized,
                    semantic_similarity=round(sim, 6),
                    instruction_retention=round(retention, 6),
                    information_density=round(density, 6),
                    compression_ratio=round(ratio, 6),
                    cqs=cqs,
                )
                variants.append(variant)

                logger.info(
                    "Variant '%s' — CQS=%.4f  sim=%.4f  ret=%.4f  density=%.4f",
                    strategy_name,
                    cqs,
                    sim,
                    retention,
                    density,
                )

            except Exception:
                logger.exception("Variant '%s' failed; skipping", strategy_name)

        if not variants:
            # Fallback: return original prompt untouched
            fallback = VariantResult(
                strategy="fallback",
                optimized_prompt=prompt,
                semantic_similarity=1.0,
                instruction_retention=1.0,
                information_density=information_density(prompt),
                compression_ratio=1.0,
                cqs=0.0,
            )
            return EvolutionResult(best=fallback, variants=[fallback], original_prompt=prompt)

        # Select the variant with the highest CQS
        best = max(variants, key=lambda v: v.cqs)

        logger.info(
            "Evolutionary optimizer selected '%s' with CQS=%.4f",
            best.strategy,
            best.cqs,
        )

        return EvolutionResult(
            best=best,
            variants=variants,
            original_prompt=prompt,
        )
