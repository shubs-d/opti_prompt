"""Instruction retention and information density metrics.

All functions are pure and stateless — no model loading required.
"""

from __future__ import annotations

import re
from typing import FrozenSet, Set

# ---------------------------------------------------------------------------
# Stop-words used to separate "meaningful" from "filler" tokens
# ---------------------------------------------------------------------------

_STOPWORDS: FrozenSet[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "am", "it", "its",
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "him", "his",
    "she", "her", "they", "them", "their", "this", "that", "these", "those",
    "of", "in", "to", "for", "on", "with", "at", "by", "from", "as",
    "if", "or", "and", "but", "so", "not", "no", "just", "very", "too",
    "also", "about", "up", "out", "into", "than", "then",
})

# Simple POS heuristic: patterns that identify "key" instruction words
_VERB_PATTERN = re.compile(
    r"^(?:write|create|generate|build|make|implement|design|develop|fix|debug|"
    r"explain|describe|summarize|analyse|analyze|compare|list|find|search|"
    r"calculate|compute|convert|extract|parse|format|validate|check|test|"
    r"deploy|install|configure|setup|run|execute|start|stop|delete|remove|"
    r"update|modify|change|add|insert|replace|merge|split|sort|filter|"
    r"read|load|save|store|send|fetch|return|show|display|render|plot|"
    r"encode|decode|encrypt|decrypt|compress|optimize|refactor|review)$",
    re.IGNORECASE,
)

_NOUN_INDICATOR = re.compile(r"^[A-Z][a-z]")  # Capitalised → likely noun
_TECHNICAL = re.compile(r"^[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)+$")  # dot-paths
_NUMERIC = re.compile(r"^\d[\d.,]*$")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _extract_key_tokens(text: str) -> Set[str]:
    """Return the set of 'important' tokens in *text*.

    A token is considered important if it is:
      - A recognised instruction verb
      - A capitalised noun
      - A technical identifier (e.g. ``os.path.join``)
      - A number
      - Any word not in the stop-word list that is ≥ 3 characters
    """
    words = re.findall(r"\S+", text.lower())
    raw_words = re.findall(r"\S+", text)  # preserve case for noun check

    key: Set[str] = set()
    for raw, lower in zip(raw_words, words):
        if _VERB_PATTERN.match(lower):
            key.add(lower)
        elif _NOUN_INDICATOR.match(raw):
            key.add(lower)
        elif _TECHNICAL.match(raw):
            key.add(lower)
        elif _NUMERIC.match(raw):
            key.add(lower)
        elif lower not in _STOPWORDS and len(lower) >= 3:
            key.add(lower)
    return key


def instruction_retention_score(original: str, optimized: str) -> float:
    """Measure what fraction of *original*'s key tokens survive in *optimized*.

    Returns:
        Float in [0, 1].  1.0 = all key tokens retained.
    """
    orig_keys = _extract_key_tokens(original)
    if not orig_keys:
        return 1.0  # nothing to lose

    opt_lower = set(re.findall(r"\S+", optimized.lower()))
    retained = orig_keys & opt_lower
    return len(retained) / len(orig_keys)


def information_density(text: str) -> float:
    """Compute the ratio of meaningful tokens to total tokens.

    ``density = meaningful_tokens / total_tokens``

    A "meaningful" token is any word not in the stop-word list.

    Returns:
        Float in [0, 1].
    """
    words = re.findall(r"\S+", text.lower())
    if not words:
        return 0.0
    meaningful = sum(1 for w in words if w not in _STOPWORDS and len(w) >= 2)
    return meaningful / len(words)
