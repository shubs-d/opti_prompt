from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_GENOME_CACHE: Optional[List[str]] = None
_GENOME_MTIME: Optional[float] = None


def _extract_rules(payload: Dict[str, Any]) -> Optional[List[str]]:
    rules = payload.get("best_genome")
    if not isinstance(rules, list):
        return None
    normalized: List[str] = [str(r) for r in rules if isinstance(r, str) and r.strip()]
    return normalized or None


def load_best_genome(path: str = "best_genome.json") -> Optional[List[str]]:
    """Load and hot-reload the best evolved genome from disk.

    The file is read only when its mtime changes. A cached in-memory genome is
    returned for repeated calls with no file modification.
    """
    global _GENOME_CACHE, _GENOME_MTIME

    file_path = Path(path)
    if not file_path.exists():
        _GENOME_CACHE = None
        _GENOME_MTIME = None
        return None

    try:
        mtime = file_path.stat().st_mtime
    except OSError:
        logger.exception("Failed to stat genome file: %s", file_path)
        return _GENOME_CACHE

    if _GENOME_CACHE is not None and _GENOME_MTIME == mtime:
        return list(_GENOME_CACHE)

    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            logger.warning("Genome file has invalid JSON structure: %s", file_path)
            return _GENOME_CACHE

        rules = _extract_rules(payload)
        if rules is None:
            logger.warning("Genome file missing valid 'best_genome': %s", file_path)
            return _GENOME_CACHE

        _GENOME_CACHE = rules
        _GENOME_MTIME = mtime
        return list(_GENOME_CACHE)

    except Exception:
        logger.exception("Failed to load genome file: %s", file_path)
        return _GENOME_CACHE
