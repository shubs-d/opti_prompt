"""Response-aware prompt evaluation.

Evaluates the responses produced by the original and optimised prompts and
computes prompt-response quality metrics that can be stored for later learning.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.core.model_loader import ModelLoader
from app.utils.similarity import cosine_similarity_score, simple_sentence_embedding

logger = logging.getLogger(__name__)


@dataclass
class ResponseMetrics:
    """Quality metrics for a single prompt/response pair."""

    semantic_coverage: float
    structural_quality: float
    length_quality: float
    information_density: float
    overall_score: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "semantic_coverage": round(self.semantic_coverage, 6),
            "structural_quality": round(self.structural_quality, 6),
            "length_quality": round(self.length_quality, 6),
            "information_density": round(self.information_density, 6),
            "overall_score": round(self.overall_score, 6),
        }


@dataclass
class ResponseEvaluationReport:
    """Comparison between original and optimised prompt responses."""

    original_response: str
    optimized_response: str
    original_metrics: ResponseMetrics
    optimized_metrics: ResponseMetrics
    improvement_score: float
    evaluation_query: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_response": self.original_response,
            "optimized_response": self.optimized_response,
            "original_metrics": self.original_metrics.to_dict(),
            "optimized_metrics": self.optimized_metrics.to_dict(),
            "improvement_score": round(self.improvement_score, 6),
            "evaluation_query": self.evaluation_query,
        }


class ResponseEvaluator:
    """Generate and score responses for prompt comparison."""

    def __init__(self, model_loader: ModelLoader) -> None:
        self._model = model_loader

    def evaluate(
        self,
        original_prompt: str,
        optimized_prompt: str,
        evaluation_query: str = "Answer the request clearly with structure and examples.",
    ) -> ResponseEvaluationReport:
        """Compare original and optimised prompt responses."""
        original_response = self._generate_response(original_prompt, evaluation_query)
        optimized_response = self._generate_response(optimized_prompt, evaluation_query)

        original_metrics = self._score_response(original_prompt, original_response)
        optimized_metrics = self._score_response(optimized_prompt, optimized_response)
        improvement = optimized_metrics.overall_score - original_metrics.overall_score

        logger.info(
            "Response evaluation — original=%.4f optimized=%.4f improvement=%.4f",
            original_metrics.overall_score,
            optimized_metrics.overall_score,
            improvement,
        )

        return ResponseEvaluationReport(
            original_response=original_response,
            optimized_response=optimized_response,
            original_metrics=original_metrics,
            optimized_metrics=optimized_metrics,
            improvement_score=improvement,
            evaluation_query=evaluation_query,
        )

    def _generate_response(self, prompt: str, evaluation_query: str) -> str:
        combined = (
            f"Instruction:\n{prompt.strip()}\n\n"
            f"Task:\n{evaluation_query.strip()}\n\n"
            "Response:\n"
        )
        generated = self._model.generate_text(combined, max_new_tokens=120, temperature=0.6)
        if generated:
            return generated
        # Conservative deterministic fallback.
        return (
            "Summary:\n"
            f"Responding to a prompt about: {prompt[:120].strip()}\n\n"
            "Key points:\n- Preserve intent\n- Explain clearly\n- Add a concrete example"
        )

    def _score_response(self, prompt: str, response: str) -> ResponseMetrics:
        prompt_embedding = simple_sentence_embedding(prompt, self._model)
        response_embedding = simple_sentence_embedding(response, self._model)
        semantic_coverage = cosine_similarity_score(prompt_embedding, response_embedding)

        structural_quality = self._score_structure(response)
        length_quality = self._score_length(response)
        information_density = self._score_information_density(response, semantic_coverage)

        overall = (
            0.40 * semantic_coverage
            + 0.20 * structural_quality
            + 0.15 * length_quality
            + 0.25 * information_density
        )

        return ResponseMetrics(
            semantic_coverage=max(0.0, min(1.0, semantic_coverage)),
            structural_quality=structural_quality,
            length_quality=length_quality,
            information_density=information_density,
            overall_score=max(0.0, min(1.0, overall)),
        )

    @staticmethod
    def _score_structure(response: str) -> float:
        sections = len(re.findall(r"^\s*(?:[-*]|\d+\.|[A-Z][A-Za-z ]+:)", response, flags=re.MULTILINE))
        paragraphs = len([p for p in response.split("\n\n") if p.strip()])
        has_colons = 1 if ":" in response else 0
        raw = min(sections * 0.2 + paragraphs * 0.15 + has_colons * 0.15, 1.0)
        return max(0.1, raw)

    def _score_length(self, response: str) -> float:
        _, tokens = self._model.tokenize(response)
        count = len(tokens)
        if 40 <= count <= 180:
            return 1.0
        if 20 <= count < 40:
            return 0.75
        if 180 < count <= 260:
            return 0.75
        if 10 <= count < 20:
            return 0.45
        return 0.35

    def _score_information_density(self, response: str, semantic_coverage: float) -> float:
        _, tokens = self._model.tokenize(response)
        token_count = max(len(tokens), 1)
        raw = semantic_coverage / token_count
        return max(0.0, min(raw * 120.0, 1.0))


def infer_prompt_structure(prompt: str) -> str:
    """Classify prompt structure for evaluation logging."""
    if re.search(r"^\s*(?:[-*]|\d+\.)", prompt, flags=re.MULTILINE):
        return "bulleted"
    if ":" in prompt or "\n\n" in prompt:
        return "sectioned"
    if len(prompt.splitlines()) > 1:
        return "multiline"
    return "plain"
