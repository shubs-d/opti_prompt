"""Token-level utility helpers."""

from __future__ import annotations

from typing import List, Tuple


def pair_tokens_with_surprisal(
    tokens: List[str],
    surprisal: List[float],
) -> List[Tuple[str, float]]:
    """Zip token strings with their surprisal scores.

    Args:
        tokens: decoded token strings.
        surprisal: per-token surprisal values (same length as *tokens*).

    Returns:
        List of (token, surprisal) tuples.
    """
    if len(tokens) != len(surprisal):
        raise ValueError(
            f"Length mismatch: {len(tokens)} tokens vs {len(surprisal)} surprisal values"
        )
    return list(zip(tokens, surprisal))


def compute_threshold(
    surprisal_values: List[float],
    aggressiveness: float,
) -> float:
    """Derive a pruning threshold from surprisal scores and aggressiveness.

    A higher *aggressiveness* (0.0–1.0) yields a higher threshold, meaning
    more tokens are considered "low-information" and eligible for removal.

    Strategy:
        threshold = mean + (1 - aggressiveness) * std

    When aggressiveness → 1.0 the threshold approaches the mean (aggressive).
    When aggressiveness → 0.0 the threshold approaches mean + std (conservative).
    """
    if not surprisal_values:
        return 0.0
    mean = sum(surprisal_values) / len(surprisal_values)
    variance = sum((v - mean) ** 2 for v in surprisal_values) / len(surprisal_values)
    std = variance ** 0.5
    return mean - aggressiveness * std


def rebuild_text_from_tokens(tokens: List[str]) -> str:
    """Reconstruct readable text from a list of sub-word tokens.

    Handles GPT-2 style 'Ġ' prefix (space marker) and SentencePiece '▁'.
    """
    parts: List[str] = []
    for tok in tokens:
        cleaned = tok.replace("Ġ", " ").replace("▁", " ")
        parts.append(cleaned)
    return "".join(parts).strip()
