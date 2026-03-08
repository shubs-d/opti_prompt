"""Diff engine — structured line-level and token-level diff output."""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class DiffResult:
    """Structured diff between an original and a compressed prompt."""

    removed: List[str] = field(default_factory=list)
    rewritten: List[Dict[str, str]] = field(default_factory=list)
    preserved: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "removed": self.removed,
            "rewritten": self.rewritten,
            "preserved": self.preserved,
        }


class DiffEngine:
    """Produces a structured diff between an original and compressed text.

    Uses both line-level and token-level comparison via Python's
    ``difflib.SequenceMatcher``.
    """

    def compute_diff(
        self,
        original: str,
        compressed: str,
    ) -> DiffResult:
        """Compare *original* and *compressed* and return a structured diff.

        Args:
            original: the full original prompt.
            compressed: the compressed prompt.

        Returns:
            A ``DiffResult`` with removed, rewritten, and preserved segments.
        """
        result = DiffResult()

        original_lines = original.splitlines(keepends=True)
        compressed_lines = compressed.splitlines(keepends=True)

        matcher = difflib.SequenceMatcher(
            isjunk=None,
            a=original_lines,
            b=compressed_lines,
            autojunk=True,
        )

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for line in original_lines[i1:i2]:
                    stripped = line.rstrip("\n\r")
                    if stripped:
                        result.preserved.append(stripped)

            elif tag == "delete":
                for line in original_lines[i1:i2]:
                    stripped = line.rstrip("\n\r")
                    if stripped:
                        result.removed.append(stripped)

            elif tag == "replace":
                orig_chunk = original_lines[i1:i2]
                comp_chunk = compressed_lines[j1:j2]
                for orig_line, comp_line in zip(orig_chunk, comp_chunk):
                    result.rewritten.append(
                        {
                            "original": orig_line.rstrip("\n\r"),
                            "compressed": comp_line.rstrip("\n\r"),
                        }
                    )
                # Handle length mismatches
                if len(orig_chunk) > len(comp_chunk):
                    for line in orig_chunk[len(comp_chunk):]:
                        stripped = line.rstrip("\n\r")
                        if stripped:
                            result.removed.append(stripped)
                elif len(comp_chunk) > len(orig_chunk):
                    for line in comp_chunk[len(orig_chunk):]:
                        stripped = line.rstrip("\n\r")
                        if stripped:
                            result.preserved.append(stripped)

            elif tag == "insert":
                # Tokens that appear only in the compressed text (rare edge
                # case) are treated as preserved.
                for line in compressed_lines[j1:j2]:
                    stripped = line.rstrip("\n\r")
                    if stripped:
                        result.preserved.append(stripped)

        logger.info(
            "Diff computed: %d removed, %d rewritten, %d preserved",
            len(result.removed),
            len(result.rewritten),
            len(result.preserved),
        )
        return result

    def compute_token_diff(
        self,
        original_tokens: List[str],
        removed_tokens: List[str],
    ) -> Dict[str, Any]:
        """Produce a token-level summary.

        Args:
            original_tokens: list of all original tokens.
            removed_tokens: list of tokens that were pruned.

        Returns:
            Dict with ``removed``, ``preserved`` token lists and counts.
        """
        removed_set = set(removed_tokens)
        preserved = [t for t in original_tokens if t not in removed_set]

        return {
            "removed_tokens": removed_tokens,
            "preserved_tokens": preserved,
            "removed_count": len(removed_tokens),
            "preserved_count": len(preserved),
        }
