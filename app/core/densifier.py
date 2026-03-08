"""Semantic densifier — phrase-level compression via rule-based rewriting.

Operates *after* token-level compression.  Loads replacement rules from
``compression_rules.json`` and applies regex substitutions to collapse
verbose patterns into semantically equivalent concise forms.

The module also performs lightweight post-processing:
  * Collapse redundant whitespace.
  * Re-capitalise sentence starts after deletions.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_RULES_PATH = Path(__file__).with_name("compression_rules.json")


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class DensificationResult:
    """Outcome of phrase-level densification."""

    densified_text: str
    original_text: str
    rules_applied: List[Dict[str, str]] = field(default_factory=list)
    phrase_reduction_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "densified_text": self.densified_text,
            "rules_applied": self.rules_applied,
            "phrase_reduction_count": self.phrase_reduction_count,
        }


# ------------------------------------------------------------------
# Rule loader
# ------------------------------------------------------------------

def _load_rules(path: Optional[Path] = None) -> List[Dict[str, str]]:
    """Load and compile compression rules from JSON.

    Each rule has ``pattern`` (regex), ``replacement``, and optional ``category``.
    """
    path = path or _RULES_PATH
    if not path.exists():
        logger.warning("Compression rules file not found at %s", path)
        return []
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    raw_rules: List[Dict[str, str]] = data.get("rules", [])
    # Pre-compile patterns for performance
    compiled: List[Dict[str, Any]] = []
    for rule in raw_rules:
        try:
            compiled.append({
                "regex": re.compile(rule["pattern"], re.IGNORECASE),
                "pattern": rule["pattern"],
                "replacement": rule["replacement"],
                "category": rule.get("category", "general"),
            })
        except re.error as exc:
            logger.warning("Skipping invalid regex '%s': %s", rule.get("pattern"), exc)
    return compiled


# Module-level cache so rules are loaded only once.
_COMPILED_RULES: Optional[List[Dict[str, Any]]] = None


def _get_rules() -> List[Dict[str, Any]]:
    global _COMPILED_RULES
    if _COMPILED_RULES is None:
        _COMPILED_RULES = _load_rules()
        logger.info("Loaded %d phrase-compression rules", len(_COMPILED_RULES))
    return _COMPILED_RULES


# ------------------------------------------------------------------
# Post-processing helpers
# ------------------------------------------------------------------

_MULTI_SPACE = re.compile(r" {2,}")
_LEADING_SPACE_AFTER_PUNCT = re.compile(r"([.!?])\s{2,}")


def _normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces, fix spacing after punctuation."""
    text = _MULTI_SPACE.sub(" ", text)
    text = _LEADING_SPACE_AFTER_PUNCT.sub(r"\1 ", text)
    return text.strip()


def _fix_capitalisation(text: str) -> str:
    """Re-capitalise the first character and sentence starts."""
    if not text:
        return text
    # Capitalise very first character
    text = text[0].upper() + text[1:]
    # Capitalise after sentence-ending punctuation
    def _cap(m: re.Match) -> str:
        return m.group(1) + " " + m.group(2).upper()
    text = re.sub(r"([.!?])\s+([a-z])", _cap, text)
    return text


# ------------------------------------------------------------------
# Densifier
# ------------------------------------------------------------------

class Densifier:
    """Phrase-level semantic densifier.

    Applies configurable regex rules to replace verbose constructions
    with shorter equivalents, then cleans up whitespace and capitalisation.
    """

    def __init__(
        self,
        rules_path: Optional[Path] = None,
        max_rules_per_pass: int = 200,
    ) -> None:
        """
        Args:
            rules_path: override path to ``compression_rules.json``.
            max_rules_per_pass: safety cap on rules applied per invocation.
        """
        if rules_path is not None:
            self._rules = _load_rules(rules_path)
        else:
            self._rules = _get_rules()
        self._max_rules = max_rules_per_pass

    def densify(
        self,
        text: str,
        intent_label: Optional[str] = None,
        aggressiveness: float = 0.5,
    ) -> DensificationResult:
        """Apply phrase-level compression to *text*.

        Args:
            text: input text (typically already token-compressed).
            intent_label: detected intent — used to skip certain rule
                categories for sensitive intents (e.g. CREATIVE).
            aggressiveness: controls which rule categories are applied.
                Low aggressiveness skips ``hedging`` and ``filler`` removals
                to preserve the author's tone.

        Returns:
            A ``DensificationResult`` with the rewritten text and metadata.
        """
        result_text = text
        applied: List[Dict[str, str]] = []
        skip_categories = self._categories_to_skip(intent_label, aggressiveness)

        for rule in self._rules:
            if len(applied) >= self._max_rules:
                break
            if rule["category"] in skip_categories:
                continue

            new_text = rule["regex"].sub(rule["replacement"], result_text)
            if new_text != result_text:
                applied.append({
                    "pattern": rule["pattern"],
                    "replacement": rule["replacement"],
                    "category": rule["category"],
                })
                result_text = new_text

        # Post-process
        result_text = _normalize_whitespace(result_text)
        result_text = _fix_capitalisation(result_text)

        logger.info(
            "Densifier applied %d rules, input %d chars → output %d chars",
            len(applied),
            len(text),
            len(result_text),
        )

        return DensificationResult(
            densified_text=result_text,
            original_text=text,
            rules_applied=applied,
            phrase_reduction_count=len(applied),
        )

    # ------------------------------------------------------------------
    # Intent-aware rule filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _categories_to_skip(
        intent_label: Optional[str],
        aggressiveness: float,
    ) -> set:
        """Determine which rule categories to skip given context.

        * CREATIVE intents skip ``hedging`` and ``filler`` (tone matters).
        * TECHNICAL intents skip ``directive_strip`` (precision matters).
        * Low aggressiveness (< 0.3) skips ``hedging`` globally.
        """
        skip: set = set()
        label = (intent_label or "").upper()

        if label == "CREATIVE":
            skip.update({"hedging", "filler", "directive_strip"})
        elif label == "TECHNICAL":
            skip.add("directive_strip")

        if aggressiveness < 0.3:
            skip.add("hedging")
            skip.add("filler")

        return skip
