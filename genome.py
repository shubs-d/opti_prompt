from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

from rules import RULE_LIBRARY, apply_rule


@dataclass(frozen=True)
class Genome:
    rules: tuple[str, ...]

    def as_key(self) -> str:
        return "|".join(self.rules)

    def compress(self, text: str, use_cache: bool = True) -> str:
        out = text
        for rule in self.rules:
            out = apply_rule(rule, out, use_cache=use_cache)
        return " ".join(out.split())



def random_genome(min_rules: int = 3, max_rules: int = 7) -> Genome:
    k = random.randint(min_rules, max_rules)
    rules = random.sample(RULE_LIBRARY, k=k)
    random.shuffle(rules)
    return Genome(tuple(rules))



def init_population(size: int, min_rules: int = 3, max_rules: int = 7) -> List[Genome]:
    return [random_genome(min_rules=min_rules, max_rules=max_rules) for _ in range(size)]



def crossover(parent_a: Genome, parent_b: Genome, min_rules: int = 3, max_rules: int = 7) -> Genome:
    a, b = list(parent_a.rules), list(parent_b.rules)
    cut_a = random.randint(1, max(1, len(a) - 1))
    cut_b = random.randint(1, max(1, len(b) - 1))
    child = a[:cut_a] + b[cut_b:]

    # Keep unique order while preserving first occurrence.
    seen = set()
    uniq = []
    for r in child:
        if r not in seen:
            seen.add(r)
            uniq.append(r)

    if len(uniq) < min_rules:
        missing = [r for r in RULE_LIBRARY if r not in seen]
        random.shuffle(missing)
        uniq.extend(missing[: min_rules - len(uniq)])

    if len(uniq) > max_rules:
        uniq = uniq[:max_rules]

    return Genome(tuple(uniq))



def mutate(genome: Genome, mutation_rate: float = 0.35, min_rules: int = 3, max_rules: int = 7) -> Genome:
    rules = list(genome.rules)
    if random.random() > mutation_rate:
        return genome

    op = random.choice(["add", "remove", "shuffle"])

    if op == "add" and len(rules) < max_rules:
        candidates = [r for r in RULE_LIBRARY if r not in rules]
        if candidates:
            insert_idx = random.randint(0, len(rules))
            rules.insert(insert_idx, random.choice(candidates))

    elif op == "remove" and len(rules) > min_rules:
        del rules[random.randrange(len(rules))]

    elif op == "shuffle" and len(rules) > 1:
        random.shuffle(rules)

    return Genome(tuple(rules))
