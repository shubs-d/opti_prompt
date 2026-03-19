"""Semantic similarity via sentence-transformers (all-MiniLM-L6-v2).

The model is loaded lazily on first call and cached as a module-level
singleton.  Embeddings are LRU-cached so repeated comparisons against the
same original prompt are essentially free.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from threading import Lock
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy singleton model holder
# ---------------------------------------------------------------------------

_model: Optional[object] = None
_model_lock = Lock()
_MODEL_NAME = "all-MiniLM-L6-v2"


def _get_model():
    """Return the cached SentenceTransformer instance."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        try:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(_MODEL_NAME)
            logger.info("Loaded sentence-transformer model '%s'", _MODEL_NAME)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed; falling back to "
                "distilgpt2 mean-pool embeddings."
            )
            _model = None
    return _model


# ---------------------------------------------------------------------------
# Cached embedding helper
# ---------------------------------------------------------------------------

@lru_cache(maxsize=512)
def _embed_text_st(text: str) -> tuple:
    """Embed *text* with sentence-transformers; returns a tuple for caching."""
    model = _get_model()
    if model is None:
        return ()
    vec: List[float] = model.encode(text, convert_to_numpy=True).tolist()
    return tuple(vec)


def _embed_text_fallback(text: str) -> List[float]:
    """Fallback: use distilgpt2 mean-pool embedding via ModelLoader."""
    from app.core.model_loader import ModelLoader

    ml = ModelLoader.get_instance()
    return ml.embed_text(text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_semantic_similarity(original: str, optimized: str) -> float:
    """Compute cosine similarity between *original* and *optimized* prompts.

    Uses ``all-MiniLM-L6-v2`` when available, otherwise falls back to the
    distilgpt2-based embeddings already present in the project.

    Returns:
        A float in [-1, 1] (typically [0, 1] for natural-language prompts).
    """
    st_model = _get_model()
    if st_model is not None:
        vec_a = _embed_text_st(original)
        vec_b = _embed_text_st(optimized)
        if vec_a and vec_b:
            a = torch.tensor(vec_a, dtype=torch.float32)
            b = torch.tensor(vec_b, dtype=torch.float32)
            if a.norm() == 0 or b.norm() == 0:
                return 0.0
            return float(
                torch.nn.functional.cosine_similarity(
                    a.unsqueeze(0), b.unsqueeze(0),
                ).item()
            )

    # Fallback path
    vec_a_fb = _embed_text_fallback(original)
    vec_b_fb = _embed_text_fallback(optimized)
    from app.utils.similarity import cosine_similarity_score

    return cosine_similarity_score(vec_a_fb, vec_b_fb)
