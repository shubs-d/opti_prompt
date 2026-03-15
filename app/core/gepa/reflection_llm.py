"""Reflection LLM abstraction used by GEPA repair.

This module provides an async interface for reflective repair:
  * Compare original prompt vs broken compressed prompt
  * Diagnose missing constraints / semantics
  * Produce a repaired prompt and concise reasoning

The default implementation uses an OpenAI-compatible Chat Completions API
when credentials are available, and falls back to a deterministic heuristic
reflection engine otherwise.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ReflectionResult:
    """Result from a reflection + repair pass."""

    repaired_prompt: str
    reasoning: str
    provider: str
    model: str
    raw: Optional[Dict[str, Any]] = None


class ReflectionLLM(ABC):
    """Async interface for reflection-based prompt repair."""

    @abstractmethod
    async def reflect(
        self,
        original_prompt: str,
        broken_prompt: str,
        mutation_hint: Optional[str] = None,
        target_token_count: Optional[int] = None,
    ) -> ReflectionResult:
        """Return a repaired prompt and natural-language diagnosis."""


class OpenAICompatibleReflectionLLM(ReflectionLLM):
    """OpenAI-compatible reflection backend.

    Uses a raw HTTP call so this project does not require an additional SDK.
    Any OpenAI-compatible endpoint can be used via environment variables.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        timeout_seconds: int = 25,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def reflect(
        self,
        original_prompt: str,
        broken_prompt: str,
        mutation_hint: Optional[str] = None,
        target_token_count: Optional[int] = None,
    ) -> ReflectionResult:
        payload = self._build_payload(
            original_prompt=original_prompt,
            broken_prompt=broken_prompt,
            mutation_hint=mutation_hint,
            target_token_count=target_token_count,
        )
        response_json = await asyncio.to_thread(self._post_json, payload)
        content = self._extract_content(response_json)
        parsed = self._parse_reflection_json(content)

        repaired = (parsed.get("repaired_prompt") or broken_prompt).strip()
        reasoning = (parsed.get("reasoning") or "No reasoning returned by reflection model.").strip()

        return ReflectionResult(
            repaired_prompt=repaired,
            reasoning=reasoning,
            provider="openai-compatible",
            model=self.model,
            raw=parsed,
        )

    def _build_payload(
        self,
        original_prompt: str,
        broken_prompt: str,
        mutation_hint: Optional[str],
        target_token_count: Optional[int],
    ) -> Dict[str, Any]:
        budget_text = (
            f"Target repaired prompt length: ~{target_token_count} tokens or fewer, "
            "without sacrificing critical intent."
            if target_token_count
            else "Keep repaired prompt concise while preserving critical intent."
        )

        hint_text = mutation_hint or "No mutation hint provided."

        system = (
            "You are a prompt repair specialist operating inside GEPA. "
            "Diagnose semantic losses introduced by compression and return strictly JSON."
        )

        user = (
            "Reflect on the semantic drift between ORIGINAL and BROKEN prompts.\n"
            f"{budget_text}\n"
            f"Mutation hint: {hint_text}\n\n"
            "Return JSON with keys:\n"
            "- reasoning: short diagnosis of what was lost\n"
            "- lost_information: list of specific missing constraints/details\n"
            "- repaired_prompt: repaired and optimized prompt\n\n"
            "ORIGINAL:\n"
            f"{original_prompt}\n\n"
            "BROKEN:\n"
            f"{broken_prompt}\n"
        )

        return {
            "model": self.model,
            "temperature": 0.4,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

    def _post_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            method="POST",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Reflection LLM HTTP error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Reflection LLM network error: {exc}") from exc

    @staticmethod
    def _extract_content(response_json: Dict[str, Any]) -> str:
        choices = response_json.get("choices") or []
        if not choices:
            return "{}"
        message = choices[0].get("message") or {}
        return str(message.get("content") or "{}")

    @staticmethod
    def _parse_reflection_json(content: str) -> Dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Some providers wrap JSON with extra text. Extract the first JSON object.
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not match:
                return {"reasoning": "Malformed reflection output.", "repaired_prompt": ""}
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {"reasoning": "Malformed reflection output.", "repaired_prompt": ""}


class HeuristicReflectionLLM(ReflectionLLM):
    """Deterministic fallback reflection backend.

    This keeps GEPA functional when external LLM credentials are absent.
    """

    def __init__(self) -> None:
        self.model = "heuristic-repair-v1"

    async def reflect(
        self,
        original_prompt: str,
        broken_prompt: str,
        mutation_hint: Optional[str] = None,
        target_token_count: Optional[int] = None,
    ) -> ReflectionResult:
        repaired, lost_chunks = self._repair_by_reinsertion(
            original_prompt=original_prompt,
            broken_prompt=broken_prompt,
            target_token_count=target_token_count,
        )
        hint = mutation_hint or "none"
        reasoning = (
            "Heuristic reflection identified likely dropped clauses and reinserted "
            f"{len(lost_chunks)} critical segment(s). Mutation hint used: {hint}."
        )

        return ReflectionResult(
            repaired_prompt=repaired,
            reasoning=reasoning,
            provider="heuristic",
            model=self.model,
            raw={"lost_information": lost_chunks},
        )

    @staticmethod
    def _repair_by_reinsertion(
        original_prompt: str,
        broken_prompt: str,
        target_token_count: Optional[int],
    ) -> tuple[str, list[str]]:
        orig_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", original_prompt) if s.strip()]
        broken_lower = broken_prompt.lower()

        missing = [s for s in orig_sentences if s.lower() not in broken_lower]
        selected = missing[:3]

        repaired = broken_prompt.strip()
        if selected:
            repaired = f"{repaired}\n\nCritical constraints to preserve:\n- " + "\n- ".join(selected)

        # Soft budget trim by words when a target count is provided.
        if target_token_count and target_token_count > 0:
            words = repaired.split()
            if len(words) > int(target_token_count * 1.15):
                words = words[: int(target_token_count * 1.15)]
                repaired = " ".join(words)

        return repaired.strip(), selected


def build_reflection_llm_from_env() -> ReflectionLLM:
    """Build the best available reflection backend from environment settings."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return OpenAICompatibleReflectionLLM(
            api_key=api_key,
            model=os.getenv("OPENAI_REFLECTION_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            timeout_seconds=int(os.getenv("OPENAI_REFLECTION_TIMEOUT", "25")),
        )
    return HeuristicReflectionLLM()
