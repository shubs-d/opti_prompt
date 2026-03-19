"""Automatic template extraction — converts optimised prompts into reusable templates.

Uses simple heuristics to detect action verbs and replace trailing noun
phrases with named placeholders.

Example::

    >>> extract_template("write a python script to sort numbers in a list")
    {"template": "write a python script to {action} {input}", "variables": ["action", "input"]}
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


# Action verbs that typically start an instruction prompt.
_ACTION_VERBS = {
    "write", "create", "build", "implement", "design", "develop",
    "generate", "explain", "describe", "summarize", "translate",
    "convert", "compute", "calculate", "analyze", "compare",
    "evaluate", "fix", "debug", "deploy", "test", "optimize",
    "list", "find", "search", "sort", "filter", "solve", "make",
    "show", "tell", "define", "refactor",
}

# Noun-like words that can be turned into variable placeholders.
# Maps common categories to their placeholder name.
_NOUN_CATEGORIES: List[Dict[str, Any]] = [
    {
        "placeholder": "language",
        "words": {
            "python", "javascript", "typescript", "java", "rust", "go",
            "c++", "c#", "ruby", "php", "swift", "kotlin", "scala",
            "html", "css", "sql", "bash", "shell", "r",
        },
    },
    {
        "placeholder": "document_type",
        "words": {
            "essay", "report", "paper", "article", "letter", "email",
            "blog", "post", "story", "poem", "script", "summary",
            "review", "proposal", "presentation", "memo", "thesis",
            "dissertation", "book", "chapter", "paragraph", "question",
        },
    },
    {
        "placeholder": "data_structure",
        "words": {
            "list", "array", "dictionary", "dict", "set", "tuple",
            "queue", "stack", "tree", "graph", "map", "table",
            "matrix", "vector", "heap", "hashmap",
        },
    },
    {
        "placeholder": "subject",
        "words": {
            "math", "science", "history", "physics", "chemistry",
            "biology", "geography", "economics", "philosophy",
            "literature", "music", "art", "engineering",
            "machine learning", "ai", "data science", "statistics",
        },
    },
]


def extract_template(prompt: str) -> Dict[str, Any]:
    """Convert an optimised prompt into a reusable template.

    Args:
        prompt: the (already optimised) prompt text.

    Returns:
        A dict with ``"template"`` (str) and ``"variables"`` (list of
        placeholder names).

    Example::

        >>> extract_template("solve this math question paper")
        {"template": "solve {input}", "variables": ["input"]}
    """
    if not prompt or not prompt.strip():
        return {"template": prompt or "", "variables": []}

    words = prompt.strip().split()
    if not words:
        return {"template": prompt, "variables": []}

    template_words: List[str] = list(words)
    variables: List[str] = []
    used_placeholders: set = set()

    # 1. Detect category-specific nouns and replace with placeholders
    for category in _NOUN_CATEGORIES:
        placeholder = category["placeholder"]
        cat_words = category["words"]

        for i, word in enumerate(template_words):
            # Skip if already a placeholder
            if word.startswith("{"):
                continue

            clean = word.lower().rstrip(".,;:!?")
            if clean in cat_words and placeholder not in used_placeholders:
                template_words[i] = "{" + placeholder + "}"
                if placeholder not in variables:
                    variables.append(placeholder)
                used_placeholders.add(placeholder)

    # 2. If no placeholders were found, try a generic fallback:
    #    replace the last noun-like span with {input}.
    if not variables and len(words) > 2:
        # Pick the last non-verb, non-preposition word
        last_word = template_words[-1].lower().rstrip(".,;:!?")
        if last_word not in _ACTION_VERBS and last_word not in _SMALL_WORDS:
            template_words[-1] = "{input}"
            variables.append("input")

    template_str = " ".join(template_words)

    return {"template": template_str, "variables": variables}


# Small / function words that should not be turned into placeholders.
_SMALL_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "to", "of", "in", "on", "at", "for", "with", "by", "from",
    "and", "or", "but", "not", "no", "so", "if", "as", "it",
    "that", "this", "how", "what", "which", "who", "where", "when",
    "up", "out", "about", "into", "through", "during", "before",
    "after", "above", "below", "between",
}
