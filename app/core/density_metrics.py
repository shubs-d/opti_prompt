"""Density metrics — information-density scoring for prompts.

The density score approximates *semantic information per token*.
A prompt that preserves the meaning of the original in fewer tokens
has higher density.

Score formula:
    density = semantic_similarity * (original_tokens / compressed_tokens)

This rewards compression that keeps similarity high while reducing
length.  The raw density is normalised to [0, 1] for cleaner
downstream consumption.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class DensityReport:
    """Results of information-density analysis."""

    density_score: float
    semantic_similarity: float
    original_token_count: int
    compressed_token_count: int
    compression_ratio: float
    information_per_token: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "density_score": round(self.density_score, 6),
            "semantic_similarity": round(self.semantic_similarity, 6),
            "original_token_count": self.original_token_count,
            "compressed_token_count": self.compressed_token_count,
            "compression_ratio": round(self.compression_ratio, 4),
            "information_per_token": round(self.information_per_token, 6),
        }


class DensityMetrics:
    """Computes information-density scores for prompt candidates.

    Uses embedding-based similarity from the existing ``similarity.py``
    combined with token count metrics.
    """

    def __init__(self, model_loader: "ModelLoader") -> None:  # noqa: F821
        self._model = model_loader

    def score(
        self,
        original_text: str,
        candidate_text: str,
    ) -> DensityReport:
        """Compute density metrics for *candidate_text* relative to *original_text*.

        Args:
            original_text: the unmodified prompt.
            candidate_text: a compressed / rewritten candidate.

        Returns:
            A ``DensityReport`` with density score and sub-metrics.
        """
        from app.utils.similarity import cosine_similarity_score, simple_sentence_embedding

        # Embedding-based semantic similarity
        orig_emb = simple_sentence_embedding(original_text, self._model)
        cand_emb = simple_sentence_embedding(candidate_text, self._model)
        similarity = cosine_similarity_score(orig_emb, cand_emb)

        # Token counts (using the model's own tokeniser for consistency)
        _, orig_tokens = self._model.tokenize(original_text)
        _, cand_tokens = self._model.tokenize(candidate_text)
        orig_count = len(orig_tokens)
        cand_count = max(len(cand_tokens), 1)  # guard div-by-zero

        compression_ratio = cand_count / orig_count if orig_count > 0 else 1.0

        # Raw density: similarity × how much shorter the candidate is.
        # When cand_count == orig_count, multiplier == 1.
        # When cand_count < orig_count, multiplier > 1 (rewarded).
        raw_density = similarity * (orig_count / cand_count)

        # Normalise to [0, 1].  In practice raw_density ∈ [0 .. ~3],
        # so we clamp with a sigmoid-style squash:  score = min(raw / 2, 1)
        density_score = min(raw_density / 2.0, 1.0)

        # Information-per-token: semantic similarity spread over tokens
        info_per_token = similarity / cand_count

        logger.info(
            "Density — sim=%.4f  ratio=%.3f  density=%.4f  ipt=%.6f",
            similarity,
            compression_ratio,
            density_score,
            info_per_token,
        )

        return DensityReport(
            density_score=density_score,
            semantic_similarity=similarity,
            original_token_count=orig_count,
            compressed_token_count=cand_count,
            compression_ratio=compression_ratio,
            information_per_token=info_per_token,
        )

    def score_candidates(
        self,
        original_text: str,
        candidates: List[str],
    ) -> List[DensityReport]:
        """Score multiple candidates at once.

        Returns a list of ``DensityReport`` objects in the same order as
        the input *candidates*.
        """
        # Pre-embed the original once
        from app.utils.similarity import cosine_similarity_score, simple_sentence_embedding

        orig_emb = simple_sentence_embedding(original_text, self._model)
        _, orig_tokens = self._model.tokenize(original_text)
        orig_count = len(orig_tokens)

        reports: List[DensityReport] = []
        for candidate in candidates:
            cand_emb = simple_sentence_embedding(candidate, self._model)
            similarity = cosine_similarity_score(orig_emb, cand_emb)
            _, cand_tokens = self._model.tokenize(candidate)
            cand_count = max(len(cand_tokens), 1)
            compression_ratio = cand_count / orig_count if orig_count > 0 else 1.0
            raw_density = similarity * (orig_count / cand_count)
            density_score = min(raw_density / 2.0, 1.0)
            info_per_token = similarity / cand_count

            reports.append(DensityReport(
                density_score=density_score,
                semantic_similarity=similarity,
                original_token_count=orig_count,
                compressed_token_count=cand_count,
                compression_ratio=compression_ratio,
                information_per_token=info_per_token,
            ))

        return reports
