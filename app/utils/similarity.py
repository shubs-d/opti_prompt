"""Similarity helpers — cosine similarity on sentence embeddings."""

from __future__ import annotations

import logging
from typing import List

import torch

logger = logging.getLogger(__name__)


def cosine_similarity_score(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec_a: first vector.
        vec_b: second vector.

    Returns:
        Cosine similarity in [-1, 1].
    """
    a = torch.tensor(vec_a, dtype=torch.float32)
    b = torch.tensor(vec_b, dtype=torch.float32)
    if a.norm() == 0 or b.norm() == 0:
        return 0.0
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def simple_sentence_embedding(
    text: str,
    model_loader: "ModelLoader",  # noqa: F821 — forward ref
) -> List[float]:
    """Produce a simple sentence embedding by mean-pooling the last hidden states.

    This avoids pulling in a dedicated sentence-transformer model; instead it
    reuses the already-loaded causal LM hidden states.

    Args:
        text: input string.
        model_loader: an initialised ``ModelLoader`` instance.

    Returns:
        A list of floats representing the embedding vector.
    """
    return model_loader.embed_text(text)
