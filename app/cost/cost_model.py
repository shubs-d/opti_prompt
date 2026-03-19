"""Token-based LLM compute cost estimator.

Uses tiktoken for accurate token counts and configurable per-model pricing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import tiktoken

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing — USD per token (input pricing)
# ---------------------------------------------------------------------------

MODEL_PRICING: Dict[str, float] = {
    "gpt-4": 0.03 / 1000,
    "gpt-3.5": 0.002 / 1000,
    "claude": 0.015 / 1000,
}

# Default model used when caller does not specify one
_DEFAULT_MODEL = "gpt-4"

# Tiktoken encoder (cl100k_base covers GPT-3.5 / GPT-4 / Claude-style)
_encoder = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    """Return the number of tokens in *text* using cl100k_base encoding."""
    return len(_encoder.encode(text))


def estimate_cost(text: str, model: str = _DEFAULT_MODEL) -> float:
    """Estimate the input cost (USD) for sending *text* to *model*.

    Args:
        text: prompt string.
        model: one of the keys in ``MODEL_PRICING``.

    Returns:
        Estimated cost in USD.
    """
    price_per_token = MODEL_PRICING.get(model, MODEL_PRICING[_DEFAULT_MODEL])
    tokens = count_tokens(text)
    return tokens * price_per_token


def compare_costs(
    original: str,
    optimized: str,
    model: str = _DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Compare costs between *original* and *optimized* prompts.

    Returns:
        Dict with ``original_cost``, ``optimized_cost``, ``savings_percent``,
        ``original_tokens``, ``optimized_tokens``, and ``model``.
    """
    orig_tokens = count_tokens(original)
    opt_tokens = count_tokens(optimized)
    price = MODEL_PRICING.get(model, MODEL_PRICING[_DEFAULT_MODEL])

    orig_cost = orig_tokens * price
    opt_cost = opt_tokens * price
    savings = ((orig_cost - opt_cost) / orig_cost * 100.0) if orig_cost > 0 else 0.0

    return {
        "original_cost": round(orig_cost, 8),
        "optimized_cost": round(opt_cost, 8),
        "savings_percent": round(savings, 4),
        "original_tokens": orig_tokens,
        "optimized_tokens": opt_tokens,
        "model": model,
    }
