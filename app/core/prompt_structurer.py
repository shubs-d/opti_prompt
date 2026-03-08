"""Prompt structurer — optionally reformats prompts into concise structured form.

Converts verbose single-sentence prompts into a *topic + aspect list* format
when doing so would increase information density without losing meaning.

Example:
    Input:   "Explain gradient descent including the intuition behind it,
              the mathematical formulation, and a practical example"
    Output:  "Explain gradient descent: intuition, mathematical formulation, example"

Heuristics for when structuring is beneficial:
  * Prompt contains enumerable sub-topics joined by connectives.
  * The prompt is a single compound sentence (no paragraph breaks).
  * The detected intent is INFORMATIONAL, TECHNICAL, or ANALYTICAL.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class StructuringResult:
    """Output of the structuring pass."""

    structured_text: str
    was_restructured: bool
    components: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "structured_text": self.structured_text,
            "was_restructured": self.was_restructured,
            "components": self.components,
        }


# ------------------------------------------------------------------
# Patterns for decomposing compound prompts
# ------------------------------------------------------------------

# Connectives that separate enumerable aspects
_SPLITTER = re.compile(
    r",\s*(?:and |as well as |along with |plus |also |including )?|"
    r"\band\b\s+|"
    r"\bas well as\b\s+|"
    r"\balong with\b\s+|"
    r"\bincluding\b\s+|"
    r"\bplus\b\s+",
    re.IGNORECASE,
)

# Leading verbs / instructions that define the *task* prefix
_TASK_PREFIX = re.compile(
    r"^(explain|describe|list|compare|analyze|analyse|discuss|outline|summarize|summarise|"
    r"write about|tell me about|give me|provide|show|detail|elaborate on|break down)\s+",
    re.IGNORECASE,
)

# Noise words that can be trimmed from extracted aspects
_NOISE = re.compile(
    r"^(the|a|an|its|their|about|of|for|in|with|how|what|some|any)\s+",
    re.IGNORECASE,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _extract_task_prefix(text: str) -> tuple[str, str]:
    """Split text into (task_verb, remainder).

    Returns ("", text) if no task verb is found.
    """
    m = _TASK_PREFIX.match(text)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    return "", text


def _extract_topic_and_aspects(remainder: str) -> tuple[str, List[str]]:
    """Attempt to separate the core topic from enumerable aspects.

    Strategy: split on connectives; treat the first chunk as the topic
    and later chunks as aspects. If the split produces only one chunk
    the prompt isn't compound → return empty aspects.
    """
    parts = _SPLITTER.split(remainder)
    parts = [p.strip().rstrip(".!?") for p in parts if p and p.strip()]

    if len(parts) < 2:
        return remainder, []

    topic = parts[0]
    aspects: List[str] = []
    for part in parts[1:]:
        # Strip leading noise words for brevity
        cleaned = _NOISE.sub("", part).strip()
        if cleaned:
            aspects.append(cleaned)

    return topic, aspects


def _should_structure(
    text: str,
    intent_label: Optional[str],
    aspects: List[str],
) -> bool:
    """Decide whether structuring is beneficial."""
    # Only structure if we found enumerable aspects
    if len(aspects) < 2:
        return False
    # Skip for creative / conversational intents (structure can hurt tone)
    label = (intent_label or "").upper()
    if label in {"CREATIVE", "CONVERSATIONAL"}:
        return False
    # Skip very short prompts (structuring adds overhead)
    if len(text) < 40:
        return False
    return True


# ------------------------------------------------------------------
# Structurer
# ------------------------------------------------------------------

class PromptStructurer:
    """Optionally restructures a prompt into *topic: aspect, aspect, …* form."""

    def structure(
        self,
        text: str,
        intent_label: Optional[str] = None,
    ) -> StructuringResult:
        """Attempt to restructure *text*.

        If restructuring isn't appropriate, returns the text unchanged with
        ``was_restructured=False``.
        """
        task_verb, remainder = _extract_task_prefix(text)
        topic, aspects = _extract_topic_and_aspects(remainder)

        if not _should_structure(text, intent_label, aspects):
            return StructuringResult(
                structured_text=text,
                was_restructured=False,
                components=[],
            )

        # Build structured form
        if task_verb:
            structured = f"{task_verb.capitalize()} {topic}: {', '.join(aspects)}"
        else:
            structured = f"{topic}: {', '.join(aspects)}"

        # Only use the restructured version if it's actually shorter
        if len(structured) >= len(text):
            return StructuringResult(
                structured_text=text,
                was_restructured=False,
                components=[],
            )

        components = [topic] + aspects
        logger.info(
            "Structured prompt into %d components (saved %d chars)",
            len(components),
            len(text) - len(structured),
        )

        return StructuringResult(
            structured_text=structured,
            was_restructured=True,
            components=components,
        )
