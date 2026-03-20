"""Application-level lightweight configuration constants."""

from __future__ import annotations

# Common conversational fillers that usually do not add instruction value.
FILLER_WORD_REGEX = (
    r"(?i)\b(?:"
    r"please|thanks?|thank\s+you|kindly|"
    r"could\s+you|can\s+you|would\s+you|"
    r"can\s+you\s+help\s+me|"
    r"i\s+would\s+like\s+to|"
    r"sorry\s+but|"
    r"act\s+as\s+a|"
    r"just\s+wondering|"
    r"if\s+you\s+don'?t\s+mind"
    r")\b"
)

# Bits of token entropy below this are considered low-information by default.
ENTROPY_THRESHOLD = 2.5

# Controlled second-stage compression window (total compression target).
# The backend enforces this range when the controlled mode is enabled.
CONTROLLED_ENFORCE_COMPRESSION_WINDOW = True
CONTROLLED_MIN_TOTAL_COMPRESSION_PERCENT = 15.0
CONTROLLED_MAX_TOTAL_COMPRESSION_PERCENT = 45.0
