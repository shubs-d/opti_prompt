"""CLI interface for the LLM Context Optimization Engine."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer

# Ensure the project root is on sys.path so ``app.*`` imports resolve
# when the CLI is invoked directly (e.g. ``python cli.py``).
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.core.compressor import Compressor
from app.core.decision_engine import DecisionEngine
from app.core.diff_engine import DiffEngine
from app.core.evaluator import Evaluator
from app.core.intent_engine import IntentEngine
from app.core.model_loader import ModelLoader

cli = typer.Typer(
    name="context-optimizer",
    help="LLM Context Optimization Engine — compress prompts via surprisal scoring.",
    add_completion=False,
)


def _run_pipeline(
    file: Path,
    aggressiveness: float,
    model_name: str,
    test_query: str,
) -> None:
    """Core optimization pipeline shared by command and callback."""
    prompt_text = file.read_text(encoding="utf-8").strip()
    if not prompt_text:
        typer.echo("Error: prompt file is empty.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Loading model '{model_name}' …")
    model_loader = ModelLoader.get_instance(model_name=model_name)

    typer.echo("Compressing …")
    compressor = Compressor(model_loader)
    compression = compressor.compress_prompt(
        text=prompt_text,
        aggressiveness=aggressiveness,
    )

    typer.echo("Computing diff …")
    diff_engine = DiffEngine()
    diff_result = diff_engine.compute_diff(
        original=prompt_text,
        compressed=compression.compressed_text,
    )

    typer.echo("Evaluating quality drift …")
    evaluator = Evaluator(model_loader)
    evaluation = evaluator.evaluate(
        original_prompt=prompt_text,
        compressed_prompt=compression.compressed_text,
        test_query=test_query,
    )

    typer.echo("Making decision …")
    decision_engine = DecisionEngine()
    decision = decision_engine.decide(
        token_reduction_percent=compression.token_reduction_percent,
        drift_score=evaluation.drift_score,
    )

    report = {
        "compressed_prompt": compression.compressed_text,
        "token_reduction_percent": round(compression.token_reduction_percent, 4),
        "diff": diff_result.to_dict(),
        "evaluation": evaluation.to_dict(),
        "decision": decision,
    }

    typer.echo("\n" + json.dumps(report, indent=2, ensure_ascii=False))


@cli.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    file: Optional[Path] = typer.Option(
        None,
        "--file",
        "-f",
        help="Path to a text file containing the prompt.",
    ),
    aggressiveness: Optional[float] = typer.Option(
        None,
        "--aggressiveness",
        "-a",
        min=0.0,
        max=1.0,
        help="Pruning intensity (0 = conservative, 1 = aggressive). Omit for auto mode.",
    ),
    auto: bool = typer.Option(
        True,
        "--auto/--no-auto",
        help="Enable/disable automatic aggressiveness based on intent detection.",
    ),
    model_name: str = typer.Option(
        "distilgpt2",
        "--model",
        "-m",
        help="Hugging Face model identifier.",
    ),
    test_query: str = typer.Option(
        "Summarize the above context.",
        "--query",
        "-q",
        help="Probe query for evaluation.",
    ),
) -> None:
    """Root callback — runs optimize when invoked without a subcommand."""
    if ctx.invoked_subcommand is not None:
        return
    if file is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()
    _run_pipeline(file, aggressiveness, model_name, test_query, auto)


@cli.command("optimize")
def optimize_cmd(
    file: Path = typer.Option(
        ...,
        "--file",
        "-f",
        exists=True,
        readable=True,
        help="Path to a text file containing the prompt.",
    ),
    aggressiveness: Optional[float] = typer.Option(
        None,
        "--aggressiveness",
        "-a",
        min=0.0,
        max=1.0,
        help="Pruning intensity (0 = conservative, 1 = aggressive). Omit for auto mode.",
    ),
    auto: bool = typer.Option(
        True,
        "--auto/--no-auto",
        help="Enable/disable automatic aggressiveness based on intent detection.",
    ),
    model_name: str = typer.Option(
        "distilgpt2",
        "--model",
        "-m",
        help="Hugging Face model identifier.",
    ),
    test_query: str = typer.Option(
        "Summarize the above context.",
        "--query",
        "-q",
        help="Probe query for evaluation.",
    ),
) -> None:
    """Compress a prompt file and print a structured JSON report."""
    _run_pipeline(file, aggressiveness, model_name, test_query, auto)


if __name__ == "__main__":
    cli()
