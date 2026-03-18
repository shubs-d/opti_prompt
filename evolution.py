from __future__ import annotations

import json
import multiprocessing as mp
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from evaluator import EvalResult, evaluate_genome_worker
from genome import Genome, crossover, init_population, mutate, random_genome


@dataclass
class EvolutionConfig:
    population_size: int = 30
    generations: int = 15
    elite_fraction: float = 0.3
    mutation_rate: float = 0.35
    min_rules: int = 3
    max_rules: int = 7
    eval_batch_size: int = 80
    use_cache: bool = True
    use_parallel: bool = True
    processes: int | None = None
    seed: int = 42
    output_path: str = "evolution_results.json"
    best_genome_path: str = "best_genome.json"



def _select_prompts(prompts: List[str], k: int, seed: int) -> List[str]:
    random.seed(seed)
    if len(prompts) <= k:
        return prompts
    return random.sample(prompts, k=k)



def _evaluate_population(
    population: List[Genome],
    prompts: List[str],
    use_cache: bool,
    use_parallel: bool,
    processes: int | None,
) -> Dict[Genome, EvalResult]:
    jobs = [(g, prompts, use_cache) for g in population]

    if use_parallel and len(population) > 1:
        with mp.Pool(processes=processes) as pool:
            results = pool.map(evaluate_genome_worker, jobs)
    else:
        results = [evaluate_genome_worker(job) for job in jobs]

    # Worker now returns (genome, result, local_cache); discard local_cache here.
    return {g: r for g, r, _cache in results}



def _tournament_select(scored: List[Tuple[Genome, EvalResult]], k: int = 3) -> Genome:
    candidates = random.sample(scored, k=min(k, len(scored)))
    candidates.sort(key=lambda x: x[1].fitness, reverse=True)
    return candidates[0][0]



def run_evolution(prompts: List[str], config: EvolutionConfig) -> Dict[str, object]:
    random.seed(config.seed)

    population = init_population(
        size=config.population_size,
        min_rules=config.min_rules,
        max_rules=config.max_rules,
    )

    best_genome: Genome | None = None
    best_result: EvalResult | None = None
    history: List[Dict[str, object]] = []

    for gen in range(config.generations):
        eval_prompts = _select_prompts(
            prompts,
            k=min(config.eval_batch_size, len(prompts)),
            seed=config.seed + gen,
        )

        score_map = _evaluate_population(
            population,
            prompts=eval_prompts,
            use_cache=config.use_cache,
            use_parallel=config.use_parallel,
            processes=config.processes,
        )

        scored = list(score_map.items())
        scored.sort(key=lambda x: x[1].fitness, reverse=True)

        gen_best_g, gen_best_r = scored[0]
        if best_result is None or gen_best_r.fitness > best_result.fitness:
            best_genome, best_result = gen_best_g, gen_best_r

        history.append(
            {
                "generation": gen,
                "best_fitness": gen_best_r.fitness,
                "best_reduction_percent": gen_best_r.reduction_percent,
                "best_drift_score": gen_best_r.drift_score,
                "best_genome": list(gen_best_g.rules),
            }
        )

        elite_n = max(2, int(config.population_size * config.elite_fraction))
        elites = [g for g, _ in scored[:elite_n]]

        next_pop: List[Genome] = elites.copy()
        while len(next_pop) < config.population_size:
            parent_a = _tournament_select(scored)
            parent_b = _tournament_select(scored)
            child = crossover(
                parent_a,
                parent_b,
                min_rules=config.min_rules,
                max_rules=config.max_rules,
            )
            child = mutate(
                child,
                mutation_rate=config.mutation_rate,
                min_rules=config.min_rules,
                max_rules=config.max_rules,
            )
            next_pop.append(child)

        if random.random() < 0.15:
            next_pop[-1] = random_genome(config.min_rules, config.max_rules)

        population = next_pop

    assert best_genome is not None and best_result is not None

    result_payload: Dict[str, object] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": {
            "population_size": config.population_size,
            "generations": config.generations,
            "elite_fraction": config.elite_fraction,
            "mutation_rate": config.mutation_rate,
            "eval_batch_size": config.eval_batch_size,
            "use_cache": config.use_cache,
            "use_parallel": config.use_parallel,
            "processes": config.processes,
        },
        "best_genome": list(best_genome.rules),
        "metrics": {
            "reduction_percent": best_result.reduction_percent,
            "drift_score": best_result.drift_score,
            "fitness": best_result.fitness,
        },
        "example": {
            "original": best_result.example_original,
            "compressed": best_result.example_compressed,
        },
        "history": history,
    }

    out_path = Path(config.output_path)
    out_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")

    best_genome_payload: Dict[str, object] = {
        "timestamp": result_payload["timestamp"],
        "best_genome": list(best_genome.rules),
        "metrics": result_payload["metrics"],
        "source": {
            "results_file": config.output_path,
            "population_size": config.population_size,
            "generations": config.generations,
            "eval_batch_size": config.eval_batch_size,
        },
    }
    best_path = Path(config.best_genome_path)
    best_path.write_text(json.dumps(best_genome_payload, indent=2), encoding="utf-8")

    return result_payload
