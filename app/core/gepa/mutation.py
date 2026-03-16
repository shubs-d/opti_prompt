"""Mutation operators for GEPA prompt evolution."""

from __future__ import annotations

import random
import re
from typing import List

from app.core.compressor import Compressor
from app.core.gepa.population import PromptCandidate

_PHRASE_REWRITE_RULES = {
    "please": "",
    "could you": "",
    "can you": "",
    "i would like you to": "",
    "make sure to": "ensure",
    "in order to": "to",
    "as much as possible": "",
    "it is important that": "",
}


class MutationEngine:
    """Applies cheap text mutations to generate offspring candidates."""

    def __init__(self, compressor: Compressor, seed: int = 13) -> None:
        self._compressor = compressor
        self._rng = random.Random(seed)

    def mutate(
        self,
        original_prompt: str,
        parent: PromptCandidate,
        next_generation: int,
    ) -> List[PromptCandidate]:
        """Generate a small offspring set from a parent candidate."""
        children: List[PromptCandidate] = []

        # 1) Aggressiveness perturbation + recompression.
        delta = self._rng.choice([-0.1, -0.05, 0.05, 0.1])
        child_aggr = min(0.95, max(0.05, parent.aggressiveness + delta))
        compressed = self._compressor.compress_prompt(original_prompt, aggressiveness=child_aggr)
        children.append(
            PromptCandidate(
                prompt=compressed.compressed_text,
                aggressiveness=child_aggr,
                threshold_scale=parent.threshold_scale,
                origin="mutate_aggressiveness",
                generation=next_generation,
            )
        )

        # 2) Phrase rewrite and fluff trimming.
        rewritten = self._rewrite_phrases(parent.prompt)
        children.append(
            PromptCandidate(
                prompt=rewritten,
                aggressiveness=parent.aggressiveness,
                threshold_scale=parent.threshold_scale,
                origin="mutate_rewrite",
                generation=next_generation,
            )
        )

        # 3) Redundancy elimination.
        deduped = self._remove_redundancy(parent.prompt)
        children.append(
            PromptCandidate(
                prompt=deduped,
                aggressiveness=parent.aggressiveness,
                threshold_scale=parent.threshold_scale,
                origin="mutate_redundancy",
                generation=next_generation,
            )
        )

        # 4) Instruction restructuring for readability.
        structured = self._restructure_instructions(parent.prompt)
        children.append(
            PromptCandidate(
                prompt=structured,
                aggressiveness=parent.aggressiveness,
                threshold_scale=parent.threshold_scale,
                origin="mutate_structure",
                generation=next_generation,
            )
        )

        return [child for child in children if child.prompt.strip()]

    @staticmethod
    def _rewrite_phrases(text: str) -> str:
        updated = text
        for source, target in _PHRASE_REWRITE_RULES.items():
            updated = re.sub(rf"\\b{re.escape(source)}\\b", target, updated, flags=re.IGNORECASE)
        updated = re.sub(r"\s{2,}", " ", updated)
        return updated.strip(" ,;:\n\t")

    @staticmethod
    def _remove_redundancy(text: str) -> str:
        sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]
        seen = set()
        unique_sentences: List[str] = []
        for sentence in sentences:
            key = re.sub(r"\W+", "", sentence.lower())
            if key and key not in seen:
                seen.add(key)
                unique_sentences.append(sentence)

        if not unique_sentences:
            return text.strip()
        return " ".join(unique_sentences)

    @staticmethod
    def _restructure_instructions(text: str) -> str:
        if "\n" in text:
            return text.strip()

        chunks = [chunk.strip() for chunk in re.split(r"[.;]\s+", text) if chunk.strip()]
        if len(chunks) < 3:
            return text.strip()

        first = chunks[0]
        bullets = "\n".join(f"- {chunk}" for chunk in chunks[1:])
        return f"{first}\n\nRequirements:\n{bullets}".strip()
