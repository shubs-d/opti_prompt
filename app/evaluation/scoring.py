"""Compression Quality Score (CQS) — composite metric for optimized prompts.

CQS = 0.4 * semantic_similarity
    + 0.2 * instruction_retention
    + 0.2 * information_density
    + 0.2 * (1 - compression_ratio)
"""

from __future__ import annotations


def compression_quality_score(
    semantic_similarity: float,
    instruction_retention: float,
    information_density: float,
    compression_ratio: float,
) -> float:
    """Compute the Compression Quality Score.

    Args:
        semantic_similarity: cosine similarity in [0, 1].
        instruction_retention: fraction of key tokens retained in [0, 1].
        information_density: meaningful-token ratio in [0, 1].
        compression_ratio: ``optimized_tokens / original_tokens`` in (0, 1].

    Returns:
        CQS value in [0, 1] (higher is better).
    """
    cqs = (
        0.4 * max(0.0, min(1.0, semantic_similarity))
        + 0.2 * max(0.0, min(1.0, instruction_retention))
        + 0.2 * max(0.0, min(1.0, information_density))
        + 0.2 * max(0.0, min(1.0, 1.0 - compression_ratio))
    )
    return round(cqs, 6)
