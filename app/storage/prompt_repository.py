"""Prompt repository — thread-safe JSON and CSV persistence layer."""

from __future__ import annotations

import csv
import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Data directory — auto-created beside the package root
# ------------------------------------------------------------------
_DATA_DIR: Path = Path(__file__).resolve().parents[2] / "data"

_JSON_FILE: Path = _DATA_DIR / "prompts.json"
_CSV_FILE: Path = _DATA_DIR / "prompts.csv"

_CSV_HEADERS: List[str] = [
    "id",
    "original_prompt",
    "compressed_prompt",
    "token_reduction_percent",
    "drift_score",
    "decision",
    "created_at",
    "version",
    "intent",
    "aggressiveness_used",
    "auto_mode",
]


def _ensure_data_dir() -> None:
    """Create the data/ directory (and files) if they don't exist."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not _JSON_FILE.exists():
        _JSON_FILE.write_text("[]", encoding="utf-8")
    if not _CSV_FILE.exists():
        with _CSV_FILE.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(_CSV_HEADERS)


# ------------------------------------------------------------------
# Record factory
# ------------------------------------------------------------------

def _build_record(
    original_prompt: str,
    compressed_prompt: str,
    token_reduction_percent: float,
    drift_score: float,
    decision: str,
    prompt_id: Optional[str] = None,
    version: int = 1,
    intent: Optional[str] = None,
    aggressiveness_used: Optional[float] = None,
    auto_mode: Optional[bool] = None,
) -> Dict[str, Any]:
    return {
        "id": prompt_id or uuid.uuid4().hex[:12],
        "original_prompt": original_prompt,
        "compressed_prompt": compressed_prompt,
        "token_reduction_percent": round(token_reduction_percent, 4),
        "drift_score": round(drift_score, 6),
        "decision": decision,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": version,
        "intent": intent,
        "aggressiveness_used": round(aggressiveness_used, 4) if aggressiveness_used is not None else None,
        "auto_mode": auto_mode,
    }


# ===================================================================
# JSON storage backend
# ===================================================================

class _JsonStorage:
    """Thread-safe, append-only JSON list storage."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        _ensure_data_dir()
        self._cache: List[Dict[str, Any]] = self._load()

    # -- private -----------------------------------------------------

    def _load(self) -> List[Dict[str, Any]]:
        try:
            data = json.loads(_JSON_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, FileNotFoundError):
            pass
        return []

    def _flush(self) -> None:
        _JSON_FILE.write_text(
            json.dumps(self._cache, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # -- public ------------------------------------------------------

    def add(self, record: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            self._cache.append(record)
            self._flush()
        logger.info("JSON — stored prompt %s (v%s)", record["id"], record["version"])
        return record

    def list_all(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._cache)

    def get_by_id(self, prompt_id: str) -> List[Dict[str, Any]]:
        """Return all versions for a given prompt ID."""
        with self._lock:
            return [r for r in self._cache if r["id"] == prompt_id]

    def next_version(self, prompt_id: str) -> int:
        versions = self.get_by_id(prompt_id)
        return max((r["version"] for r in versions), default=0) + 1


# ===================================================================
# CSV storage backend
# ===================================================================

class _CsvStorage:
    """Thread-safe, append-only CSV storage."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        _ensure_data_dir()

    def add(self, record: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            with _CSV_FILE.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_CSV_HEADERS)
                writer.writerow({k: record.get(k, "") for k in _CSV_HEADERS})
        logger.info("CSV  — stored prompt %s (v%s)", record["id"], record["version"])
        return record

    def list_all(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows: List[Dict[str, Any]] = []
            try:
                with _CSV_FILE.open("r", newline="", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        # coerce types
                        row["token_reduction_percent"] = float(row.get("token_reduction_percent", 0))
                        row["drift_score"] = float(row.get("drift_score", 0))
                        row["version"] = int(row.get("version", 1))
                        # New optional fields — graceful fallback for older rows
                        row.setdefault("intent", None)
                        aggr = row.get("aggressiveness_used")
                        row["aggressiveness_used"] = float(aggr) if aggr not in (None, "") else None
                        am = row.get("auto_mode")
                        row["auto_mode"] = am.lower() == "true" if isinstance(am, str) and am else None
                        rows.append(row)
            except FileNotFoundError:
                pass
            return rows

    def get_by_id(self, prompt_id: str) -> List[Dict[str, Any]]:
        return [r for r in self.list_all() if r["id"] == prompt_id]

    def next_version(self, prompt_id: str) -> int:
        versions = self.get_by_id(prompt_id)
        return max((r["version"] for r in versions), default=0) + 1


# ===================================================================
# Public facade — selected via STORAGE_MODE env var
# ===================================================================

import os

STORAGE_MODE: str = os.getenv("STORAGE_MODE", "json").lower()


class PromptRepository:
    """Unified repository that delegates to the configured backend.

    Reads ``STORAGE_MODE`` env var (``json`` | ``csv``).  Defaults to JSON.
    Both backends are always written to so that exports stay in sync.
    """

    _instance: Optional["PromptRepository"] = None

    def __init__(self) -> None:
        self._json = _JsonStorage()
        self._csv = _CsvStorage()
        self._primary = self._json if STORAGE_MODE == "json" else self._csv
        logger.info("PromptRepository initialised — primary backend: %s", STORAGE_MODE)

    @classmethod
    def get_instance(cls) -> "PromptRepository":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # -- write -------------------------------------------------------

    def store(
        self,
        original_prompt: str,
        compressed_prompt: str,
        token_reduction_percent: float,
        drift_score: float,
        decision: str,
        prompt_id: Optional[str] = None,
        intent: Optional[str] = None,
        aggressiveness_used: Optional[float] = None,
        auto_mode: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Create a new versioned record and persist to both backends."""
        pid = prompt_id or uuid.uuid4().hex[:12]
        version = self._json.next_version(pid)

        record = _build_record(
            original_prompt=original_prompt,
            compressed_prompt=compressed_prompt,
            token_reduction_percent=token_reduction_percent,
            drift_score=drift_score,
            decision=decision,
            prompt_id=pid,
            version=version,
            intent=intent,
            aggressiveness_used=aggressiveness_used,
            auto_mode=auto_mode,
        )

        self._json.add(record)
        self._csv.add(record)
        return record

    # -- read --------------------------------------------------------

    def list_all(self) -> List[Dict[str, Any]]:
        """Return every stored record from the primary backend."""
        return self._primary.list_all()

    def get_by_id(self, prompt_id: str) -> List[Dict[str, Any]]:
        """Return all versions for *prompt_id*."""
        return self._primary.get_by_id(prompt_id)
