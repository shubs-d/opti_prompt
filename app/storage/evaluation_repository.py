"""Storage for response-aware evaluation records."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_EVAL_JSON = _DATA_DIR / "prompt_evaluations.json"


class EvaluationRepository:
    """Simple append-only JSON storage for evaluation outcomes."""

    _instance: Optional["EvaluationRepository"] = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not _EVAL_JSON.exists():
            _EVAL_JSON.write_text("[]", encoding="utf-8")
        self._cache: List[Dict[str, Any]] = self._load()

    @classmethod
    def get_instance(cls) -> "EvaluationRepository":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self) -> List[Dict[str, Any]]:
        try:
            data = json.loads(_EVAL_JSON.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, FileNotFoundError):
            pass
        return []

    def _flush(self) -> None:
        _EVAL_JSON.write_text(json.dumps(self._cache, indent=2, ensure_ascii=False), encoding="utf-8")

    def store(self, record: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if "id" not in record:
                record["id"] = uuid.uuid4().hex[:12]
            record.setdefault("created_at", datetime.now(timezone.utc).isoformat())
            self._cache.append(record)
            self._flush()
        return record

    def list_all(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._cache)
