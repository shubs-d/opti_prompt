from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

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



def _token_count(text: str) -> int:
    return max(1, len(text.split()))



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



def _drift_score(original: str, compressed: str) -> float:
    jaccard = _jaccard_similarity(original, compressed)

    ent_o = _extract_entities(original)
    ent_c = _extract_entities(compressed)
    num_o = _extract_numbers(original)
    num_c = _extract_numbers(compressed)

    ent_pres = 1.0 if not ent_o else len(ent_o & ent_c) / len(ent_o)
    num_pres = 1.0 if not num_o else len(num_o & num_c) / len(num_o)

    # Drift is lower when lexical overlap and key-token preservation are high.
    similarity = 0.55 * jaccard + 0.30 * ent_pres + 0.15 * num_pres
    return max(0.0, min(1.0, 1.0 - similarity))



def _pair_key(prompt: str, genome_key: str) -> str:
    return f"{hash(prompt)}::{genome_key}"



def compress_with_cache(genome: Genome, prompt: str, use_cache: bool = True) -> str:
    if not use_cache:
        return genome.compress(prompt, use_cache=False)

    key = _pair_key(prompt, genome.as_key())
    if key in GLOBAL_PAIR_CACHE:
        return GLOBAL_PAIR_CACHE[key]

    compressed = genome.compress(prompt, use_cache=True)
    GLOBAL_PAIR_CACHE[key] = compressed
    return compressed



def evaluate_genome(genome: Genome, prompts: List[str], use_cache: bool = True) -> EvalResult:
    reductions: List[float] = []
    drifts: List[float] = []
    penalties: List[float] = []

    ex_orig = prompts[0]
    ex_comp = ""

    for i, prompt in enumerate(prompts):
        compressed = compress_with_cache(genome, prompt, use_cache=use_cache)
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

    return EvalResult(
        reduction_percent=100.0 * reduction_mean,
        drift_score=drift_mean,
        structure_penalty=penalty_mean,
        fitness=fitness,
        example_original=ex_orig,
        example_compressed=ex_comp,
    )



def evaluate_genome_worker(args: Tuple[Genome, List[str], bool]) -> Tuple[Genome, EvalResult]:
    genome, prompts, use_cache = args
    result = evaluate_genome(genome, prompts=prompts, use_cache=use_cache)
    return genome, result
