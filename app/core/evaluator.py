"""Evaluation engine — measures quality drift after compression."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

from app.core.model_loader import ModelLoader
from app.utils.similarity import cosine_similarity_score, simple_sentence_embedding

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    """Structured quality-drift evaluation report."""

    semantic_similarity: float
    length_difference: int
    length_ratio: float
    drift_score: float
    original_response: str
    compressed_response: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "semantic_similarity": round(self.semantic_similarity, 6),
            "length_difference": self.length_difference,
            "length_ratio": round(self.length_ratio, 4),
            "drift_score": round(self.drift_score, 6),
            "original_response": self.original_response,
            "compressed_response": self.compressed_response,
        }


class Evaluator:
    """Evaluates how much quality is lost when a prompt is compressed.

    Given the original prompt, compressed prompt, and a test query the
    engine:

    1. Simulates LLM responses for both prompts (stub).
    2. Computes cosine similarity between the response embeddings.
    3. Measures length difference.
    4. Produces a composite drift score in [0, 1]  (0 = no drift).
    """

    def __init__(self, model_loader: ModelLoader) -> None:
        self._model = model_loader

    def evaluate(
        self,
        original_prompt: str,
        compressed_prompt: str,
        test_query: str = "Summarize the above context.",
    ) -> EvaluationReport:
        """Run the evaluation pipeline.

        Args:
            original_prompt: the unmodified prompt.
            compressed_prompt: the compressed prompt.
            test_query: a probe question used for simulated LLM calls.

        Returns:
            An ``EvaluationReport`` with similarity, length, and drift metrics.
        """
        # 1. Simulate LLM responses (stubs)
        original_response = self._simulate_llm_response(original_prompt, test_query)
        compressed_response = self._simulate_llm_response(compressed_prompt, test_query)

        # 2. Embed the *prompts* themselves for semantic similarity
        orig_embedding = simple_sentence_embedding(original_prompt, self._model)
        comp_embedding = simple_sentence_embedding(compressed_prompt, self._model)

        similarity = cosine_similarity_score(orig_embedding, comp_embedding)

        # 3. Length metrics
        len_orig = len(original_prompt)
        len_comp = len(compressed_prompt)
        length_diff = len_orig - len_comp
        length_ratio = len_comp / len_orig if len_orig > 0 else 1.0

        # 4. Drift score: 0 = identical, 1 = completely different
        drift = 1.0 - similarity

        logger.info(
            "Evaluation — similarity=%.4f  drift=%.4f  len_diff=%d",
            similarity,
            drift,
            length_diff,
        )

        return EvaluationReport(
            semantic_similarity=similarity,
            length_difference=length_diff,
            length_ratio=length_ratio,
            drift_score=drift,
            original_response=original_response,
            compressed_response=compressed_response,
        )

    # ------------------------------------------------------------------
    # Stub LLM call
    # ------------------------------------------------------------------

    @staticmethod
    def _simulate_llm_response(prompt: str, query: str) -> str:
        """Placeholder LLM response simulation.

        In production this would call an actual LLM endpoint.  For now
        it returns a deterministic stand-in derived from the prompt so
        that the rest of the pipeline can operate end-to-end.
        """
        word_count = len(prompt.split())
        return (
            f"[Simulated response for query '{query}' "
            f"given a {word_count}-word prompt.]"
        )
