from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import tiktoken

from genome import Genome


@dataclass
class EvalResult:
    reduction_percent: float
    drift_score: float
    structure_penalty: float
    fitness: float
    example_original: str
    example_compressed: str


GLOBAL_PAIR_CACHE: Dict[str, str] = {}



TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return max(1, len(TOKENIZER.encode(text)))



def _extract_entities(text: str) -> set[str]:
    # Lightweight entity approximation: capitalized multi-char tokens.
    return {w for w in re.findall(r"\b[A-Z][a-zA-Z0-9]{2,}\b", text)}



def _extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", text))



def _jaccard_similarity(a: str, b: str) -> float:
    sa = set(re.findall(r"\b\w+\b", a.lower()))
    sb = set(re.findall(r"\b\w+\b", b.lower()))
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = max(1, len(sa | sb))
    return inter / union



def _structure_penalty(original: str, compressed: str) -> float:
    penalty = 0.0
    if original.count("\n") >= 2 and compressed.count("\n") == 0:
        penalty += 0.15
    if len(compressed.strip()) < 40:
        penalty += 0.25
    if compressed.count("(") != compressed.count(")"):
        penalty += 0.10
    return min(1.0, penalty)



def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _set_pr_f1(original_set: set[str], compressed_set: set[str]) -> float:
    """Return F1 score for set overlap (defaults to 1.0 when both sets empty)."""
    if not original_set and not compressed_set:
        return 1.0
    overlap = len(original_set & compressed_set)
    recall = overlap / len(original_set) if original_set else 0.0
    precision = overlap / len(compressed_set) if compressed_set else 0.0
    return _f1(precision, recall)


def _drift_score(original: str, compressed: str) -> float:
    jaccard = _jaccard_similarity(original, compressed)

    ent_o = _extract_entities(original)
    ent_c = _extract_entities(compressed)
    num_o = _extract_numbers(original)
    num_c = _extract_numbers(compressed)

    ent_f1 = _set_pr_f1(ent_o, ent_c)
    num_f1 = _set_pr_f1(num_o, num_c)

    # Drift is lower when lexical overlap and key-token preservation are high.
    similarity = 0.55 * jaccard + 0.30 * ent_f1 + 0.15 * num_f1
    return max(0.0, min(1.0, 1.0 - similarity))



def _pair_key(prompt: str, genome_key: str) -> str:
    return f"{hash(prompt)}::{genome_key}"



def compress_with_cache(
    genome: Genome, prompt: str, use_cache: bool = True,
) -> Tuple[str, bool]:
    """Return (compressed_text, is_new) — is_new is True when a fresh compression was performed."""
    if not use_cache:
        return genome.compress(prompt, use_cache=False), True

    key = _pair_key(prompt, genome.as_key())
    if key in GLOBAL_PAIR_CACHE:
        return GLOBAL_PAIR_CACHE[key], False

    compressed = genome.compress(prompt, use_cache=True)
    GLOBAL_PAIR_CACHE[key] = compressed
    return compressed, True



def evaluate_genome(
    genome: Genome, prompts: List[str], use_cache: bool = True,
) -> Tuple[EvalResult, Dict[str, str]]:
    reductions: List[float] = []
    drifts: List[float] = []
    penalties: List[float] = []
    local_new_cache: Dict[str, str] = {}

    ex_orig = prompts[0]
    ex_comp = ""

    for i, prompt in enumerate(prompts):
        compressed, is_new = compress_with_cache(genome, prompt, use_cache=use_cache)
        if is_new:
            key = _pair_key(prompt, genome.as_key())
            local_new_cache[key] = compressed
        if i == 0:
            ex_comp = compressed

        orig_tokens = _token_count(prompt)
        comp_tokens = _token_count(compressed)
        reduction = max(0.0, min(1.0, (orig_tokens - comp_tokens) / orig_tokens))

        drift = _drift_score(prompt, compressed)
        penalty = _structure_penalty(prompt, compressed)

        reductions.append(reduction)
        drifts.append(drift)
        penalties.append(penalty)

    reduction_mean = sum(reductions) / max(1, len(reductions))
    drift_mean = sum(drifts) / max(1, len(drifts))
    penalty_mean = sum(penalties) / max(1, len(penalties))

    # Fitness aggressively prioritizes compression while maintaining acceptable drift.
    # Higher reduction weight (2.5 vs 1.4), lower drift penalty (0.6 vs 1.1).
    fitness = (2.5 * reduction_mean) - (0.6 * drift_mean) - (0.4 * penalty_mean)
    # Gentle smoothing keeps values stable across small sample variance.
    fitness = math.tanh(fitness)

    result = EvalResult(
        reduction_percent=100.0 * reduction_mean,
        drift_score=drift_mean,
        structure_penalty=penalty_mean,
        fitness=fitness,
        example_original=ex_orig,
        example_compressed=ex_comp,
    )
    return result, local_new_cache



def evaluate_genome_worker(
    args: Tuple[Genome, List[str], bool],
) -> Tuple[Genome, EvalResult, Dict[str, str]]:
    genome, prompts, use_cache = args
    result, local_cache = evaluate_genome(genome, prompts=prompts, use_cache=use_cache)
    return genome, result, local_cache
