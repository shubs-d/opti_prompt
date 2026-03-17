from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from typing import Any, Dict

from dataset_loader import DatasetConfig, load_prompts
from evolution import EvolutionConfig, run_evolution



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evolutionary optimization of rule-based prompt compression strategies"
    )

    parser.add_argument("--population-size", type=int, default=30)
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--eval-batch-size", type=int, default=80)
    parser.add_argument("--min-prompts", type=int, default=500)
    parser.add_argument("--target-words", type=int, default=500)
    parser.add_argument("--dataset-name", type=str, default="wikitext")
    parser.add_argument("--dataset-config", type=str, default="wikitext-103-raw-v1")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="evolution_results.json")
    parser.add_argument("--processes", type=int, default=None)

    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Disable prompt/genome and rule-level caching",
    )
    parser.add_argument(
        "--disable-parallel",
        action="store_true",
        help="Disable multiprocessing and evaluate sequentially",
    )

    return parser.parse_args()



def print_summary(result: Dict[str, Any]) -> None:
    print("Best genome:", result["best_genome"])
    print("Best compression %:", round(result["metrics"]["reduction_percent"], 4))
    print("Drift score:", round(result["metrics"]["drift_score"], 6))
    print("Fitness:", round(result["metrics"]["fitness"], 6))
    print("Example compressed prompt:\n")
    print(result["example"]["compressed"][:1200])



def main() -> None:
    args = parse_args()

    ds_cfg = DatasetConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.dataset_split,
        min_prompts=args.min_prompts,
        target_words=args.target_words,
        seed=args.seed,
    )
    prompts = load_prompts(ds_cfg)

    evo_cfg = EvolutionConfig(
        population_size=args.population_size,
        generations=args.generations,
        eval_batch_size=args.eval_batch_size,
        use_cache=not args.disable_cache,
        use_parallel=not args.disable_parallel,
        processes=args.processes,
        seed=args.seed,
        output_path=args.output,
        min_rules=4,
        max_rules=9,
    )

    result = run_evolution(prompts, evo_cfg)
    print_summary(result)

    # Also print a compact machine-readable line for scripting.
    print("\nRESULT_JSON:")
    print(json.dumps({
        "reduction_percent": result["metrics"]["reduction_percent"],
        "drift_score": result["metrics"]["drift_score"],
        "fitness": result["metrics"]["fitness"],
    }))


if __name__ == "__main__":
    mp.freeze_support()
    main()
