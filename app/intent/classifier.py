"""Lightweight 4-class prompt intent classifier.

Classes: ``coding``, ``creative``, ``qa``, ``general``.

Uses keyword heuristics for zero-latency classification — no model loading.
Each class has an associated GEPA pruning threshold and a set of token-types
that should be preserved during pruning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Set

# ---------------------------------------------------------------------------
# Keyword banks
# ---------------------------------------------------------------------------

_CODING_KEYWORDS: FrozenSet[str] = frozenset({
    "code", "function", "class", "method", "api", "bug", "debug", "error",
    "script", "program", "algorithm", "variable", "loop", "array", "list",
    "dict", "dictionary", "string", "integer", "float", "boolean", "type",
    "import", "module", "package", "library", "framework", "database", "sql",
    "query", "json", "xml", "html", "css", "javascript", "python", "java",
    "rust", "golang", "typescript", "react", "node", "docker", "kubernetes",
    "git", "compile", "runtime", "syntax", "exception", "stack", "heap",
    "pointer", "memory", "thread", "async", "await", "promise", "callback",
    "regex", "parse", "serialize", "deserialize", "endpoint", "http", "rest",
    "graphql", "implement", "refactor", "deploy", "test", "unittest",
})

_CREATIVE_KEYWORDS: FrozenSet[str] = frozenset({
    "write", "story", "poem", "essay", "creative", "imagine", "fiction",
    "narrative", "character", "plot", "dialogue", "scene", "describe",
    "vivid", "emotional", "lyric", "song", "screenplay", "novel",
    "metaphor", "simile", "tone", "voice", "style", "artistic",
    "brainstorm", "idea", "inspiration", "compose", "draft", "rewrite",
    "blog", "article", "content", "slogan", "tagline", "headline",
})

_QA_KEYWORDS: FrozenSet[str] = frozenset({
    "what", "why", "how", "when", "where", "who", "which", "explain",
    "define", "difference", "compare", "versus", "meaning", "example",
    "cause", "reason", "purpose", "pros", "cons", "advantage",
    "disadvantage", "benefit", "risk", "summary", "summarize",
    "overview", "brief", "short", "answer", "question", "does",
})

# ---------------------------------------------------------------------------
# Threshold and preservation config per intent
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IntentConfig:
    """GEPA pruning parameters for an intent class."""

    label: str
    threshold_level: str        # symbolic name: LOW / VERY_LOW / MEDIUM / HIGH
    aggressiveness: float       # numeric [0–1] for existing pipeline compat
    preserve_token_types: FrozenSet[str]  # regex tags for tokens to protect


INTENT_THRESHOLDS: Dict[str, IntentConfig] = {
    "coding": IntentConfig(
        label="coding",
        threshold_level="LOW",
        aggressiveness=0.25,
        preserve_token_types=frozenset({
            "SYNTAX", "KEYWORD", "IDENTIFIER", "OPERATOR", "BRACKET",
        }),
    ),
    "creative": IntentConfig(
        label="creative",
        threshold_level="VERY_LOW",
        aggressiveness=0.15,
        preserve_token_types=frozenset({
            "ADJECTIVE", "ADVERB", "METAPHOR",
        }),
    ),
    "qa": IntentConfig(
        label="qa",
        threshold_level="MEDIUM",
        aggressiveness=0.45,
        preserve_token_types=frozenset({
            "QUESTION_WORD", "NOUN",
        }),
    ),
    "general": IntentConfig(
        label="general",
        threshold_level="HIGH",
        aggressiveness=0.60,
        preserve_token_types=frozenset(),
    ),
}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def _score_keywords(text: str, keywords: FrozenSet[str]) -> int:
    """Count how many keywords from *keywords* appear in *text*."""
    words: Set[str] = set(re.findall(r"[a-z][a-z0-9_]+", text.lower()))
    return len(words & keywords)


def classify_intent(prompt: str) -> IntentConfig:
    """Classify *prompt* into one of four intent classes.

    Returns:
        The ``IntentConfig`` for the detected intent.
    """
    scores: List[tuple] = [
        (_score_keywords(prompt, _CODING_KEYWORDS), "coding"),
        (_score_keywords(prompt, _CREATIVE_KEYWORDS), "creative"),
        (_score_keywords(prompt, _QA_KEYWORDS), "qa"),
    ]

    # Sort by score descending; fall back to "general" if no strong signal
    scores.sort(key=lambda x: x[0], reverse=True)
    best_score, best_label = scores[0]

    if best_score < 2:
        return INTENT_THRESHOLDS["general"]

    return INTENT_THRESHOLDS[best_label]
