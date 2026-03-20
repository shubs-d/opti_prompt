from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class DatasetConfig:
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    split: str = "train"
    min_prompts: int = 500
    target_words: int = 500
    tolerance: int = 80
    max_source_rows: int = 50000
    seed: int = 42


def _word_count(text: str) -> int:
    return len(text.split())


def _clean_text(text: str) -> str:
    return " ".join(text.split())


def _chunk_to_target_words(text: str, target_words: int, rng: random.Random) -> Optional[str]:
    words = text.split()
    if len(words) < target_words:
        return None

    start_max = max(0, len(words) - target_words)
    start = rng.randint(0, start_max)
    chunk = words[start : start + target_words]
    return " ".join(chunk)


def _synthetic_paragraph(topic: str, words: int, rng: random.Random) -> str:
    sentence_pool = [
        f"This section analyzes {topic} from technical, social, and operational angles.",
        "The objective is to provide practical guidance with measurable outcomes and clear tradeoffs.",
        "Stakeholders include end users, implementers, evaluators, and policy owners with distinct incentives.",
        "Assumptions are documented so later changes can be tested against a stable baseline.",
        "Historical context matters because prior constraints often shape present architecture decisions.",
        "Quantitative metrics are useful, but qualitative signals can reveal hidden failure modes.",
        "A robust plan defines scope boundaries, escalation paths, and explicit non-goals.",
        "Execution quality depends on sequencing, communication clarity, and disciplined iteration.",
        "Risks should be prioritized by impact, reversibility, and detectability rather than by intuition alone.",
        "Experiments should be reproducible, with data provenance and parameter traces retained for audits.",
        "Counterexamples should be actively sought to reduce confirmation bias in conclusions.",
        "Implementation details should remain adaptable while preserving invariants and interfaces.",
        "Documentation quality directly affects onboarding speed and incident recovery performance.",
        "When uncertainty is high, phased rollouts and feature flags reduce systemic blast radius.",
        "Retrospectives should convert observations into specific actions with owners and deadlines.",
    ]

    result: List[str] = []
    current_words = 0
    while current_words < words:
        sentence = rng.choice(sentence_pool)
        result.append(sentence)
        current_words += _word_count(sentence)
        
    out = " ".join(result)
    return " ".join(out.split()[:words])


def _build_synthetic_prompts(min_prompts: int, target_words: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    topics = [
        "distributed systems",
        "scientific writing",
        "policy design",
        "education technology",
        "climate adaptation",
        "software reliability",
        "team collaboration",
        "medical triage",
        "public infrastructure",
        "supply chain planning",
    ]
    return [_synthetic_paragraph(rng.choice(topics), target_words, rng) for _ in range(min_prompts)]


def load_prompts(config: DatasetConfig | None = None) -> List[str]:
    """Load long-form prompts using Hugging Face datasets with synthetic fallback."""
    cfg = config or DatasetConfig()
    rng = random.Random(cfg.seed)

    try:
        from datasets import load_dataset

        dataset = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.split)
        collected: List[str] = []

        for i, row in enumerate(dataset):
            if i >= cfg.max_source_rows:
                break

            text = row.get("text") if isinstance(row, dict) else None
            if not text or not isinstance(text, str):
                continue

            text = _clean_text(text)
            if not text:
                continue

            chunk = _chunk_to_target_words(text, cfg.target_words, rng)
            if not chunk:
                continue

            collected.append(chunk)

            if len(collected) >= cfg.min_prompts:
                break

        if len(collected) >= cfg.min_prompts:
            return collected

        needed = cfg.min_prompts - len(collected)
        collected.extend(_build_synthetic_prompts(needed, cfg.target_words, cfg.seed + 7))
        return collected

    except Exception:
        # Offline or missing datasets package: use synthetic prompts to keep pipeline runnable.
        return _build_synthetic_prompts(cfg.min_prompts, cfg.target_words, cfg.seed)
