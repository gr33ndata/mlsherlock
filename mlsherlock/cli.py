"""CLI entry point: `mlsh train`."""
from __future__ import annotations

import os
import sys

import click
from dotenv import load_dotenv

load_dotenv()


_TARGET_HINTS = [
    "target", "label", "labels", "class", "classes", "output",
    "y", "survived", "price", "salary", "churn", "fraud", "diagnosis",
    "result", "outcome", "response", "dependent",
]


def _infer_task(csv_path: str, target_col: str, console: "Console") -> str:  # type: ignore[name-defined]
    """Infer classification vs regression from the target column."""
    try:
        import csv
        values = []
        with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 500:  # sample up to 500 rows
                    break
                v = row.get(target_col, "").strip()
                if v:
                    values.append(v)
    except Exception:
        return "classification"

    if not values:
        return "classification"

    unique = set(values)

    # Try parsing as floats
    try:
        numeric = [float(v) for v in values]
        unique_numeric = set(numeric)
        # Binary or few distinct integers → classification
        if len(unique_numeric) <= 2:
            task = "classification"
        elif all(float(v) == int(float(v)) for v in unique) and len(unique_numeric) <= 20:
            task = "classification"
        else:
            task = "regression"
    except ValueError:
        # Non-numeric → classification
        task = "classification"

    console.print(f"  [dim]No --task given — inferred: [bold]{task}[/bold] ({len(unique)} unique target values)[/dim]")
    return task


def _infer_target(csv_path: str, console: "Console") -> str | None:  # type: ignore[name-defined]
    """Peek at the CSV header and pick the most likely target column."""
    try:
        with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
            header = f.readline().strip().split(",")
        cols = [c.strip().strip('"').strip("'") for c in header]
    except Exception:
        return None

    if not cols:
        return None

    # 1. Exact match against known target names (case-insensitive)
    lower = [c.lower() for c in cols]
    for hint in _TARGET_HINTS:
        if hint in lower:
            inferred = cols[lower.index(hint)]
            console.print(f"  [dim]No --target given — inferred: [bold]{inferred}[/bold][/dim]")
            return inferred

    # 2. Fall back to last column (very common convention)
    inferred = cols[-1]
    console.print(f"  [dim]No --target given — using last column: [bold]{inferred}[/bold][/dim]")
    return inferred


@click.group()
def main() -> None:
    """mlsherlock — an intelligent ML agent that diagnoses and improves your models."""


@main.command()
@click.option(
    "--provider",
    default="openai",
    show_default=True,
    type=click.Choice(["anthropic", "openai"], case_sensitive=False),
    help="LLM provider to use.",
)
@click.option(
    "--data",
    default=None,
    help=(
        "Path to a local CSV file, OR a named dataset to download "
        "('titanic', 'iris', 'penguins', 'tips'), OR a Kaggle slug ('owner/dataset-name'). "
        "If omitted, the agent will ask you for a data source."
    ),
)
@click.option("--target", default=None, help="Target column name. Required when --data is a local file.")
@click.option(
    "--task",
    default=None,
    type=click.Choice(["classification", "regression"], case_sensitive=False),
    help="ML task type. Inferred from target column if not given.",
)
@click.option("--output-dir", default="./output", show_default=True, help="Directory for model and plots.")
@click.option("--max-iterations", default=20, show_default=True, type=int, help="Max agent iterations.")
@click.option(
    "--non-interactive",
    is_flag=True,
    default=False,
    help="Auto-approve all ask_user calls (picks first option).",
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    default=False,
    help="Show full agent reasoning and tool output.",
)
def train(
    provider: str,
    data: str | None,
    target: str | None,
    task: str,
    output_dir: str,
    max_iterations: int,
    non_interactive: bool,
    verbose: bool,
) -> None:
    """Train and iteratively improve a model.

    Examples:\n
      mlsh train --data titanic --target survived\n
      mlsh train --data path/to/data.csv --target price --task regression\n
      mlsh train --data owner/dataset --target label  (Kaggle)
    """
    from rich.console import Console
    from rich.rule import Rule

    console = Console()

    # Validate API key early
    if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[bold red]Error:[/bold red] ANTHROPIC_API_KEY environment variable not set.")
        console.print("Create a .env file with ANTHROPIC_API_KEY=your_key or export it in your shell.")
        sys.exit(1)
    if provider == "openai" and not (
        os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
    ):
        console.print("[bold red]Error:[/bold red] OPENAI_API_KEY (or OPENAI_KEY) environment variable not set.")
        console.print("Create a .env file with OPENAI_API_KEY=your_key or export it in your shell.")
        sys.exit(1)

    # Resolve data source and initial message for the agent
    data_path = ""      # empty = agent will download it
    data_source = ""    # what to tell the agent

    _NAMED_DATASETS = {"titanic", "iris", "penguins", "diamonds", "tips"}

    if data is None:
        # No --data given: agent will ask interactively
        data_source = "(none — agent will ask)"
        if not task:
            task = "classification"
    elif os.path.exists(data):
        # Local file path
        data_path = os.path.abspath(data)
        data_source = data_path
        if not target:
            target = _infer_target(data_path, console)
            if not target:
                console.print("[bold red]Error:[/bold red] Could not infer target column. Pass --target explicitly.")
                sys.exit(1)
        if not task:
            task = _infer_task(data_path, target, console)
    elif data in _NAMED_DATASETS or ("/" in data and not data.startswith("/")):
        # Named dataset or Kaggle slug — agent will download it
        data_source = data
        if not task:
            task = "classification"
    else:
        console.print(f"[bold red]Error:[/bold red] File not found and not a known dataset: {data!r}")
        console.print("Pass a local CSV path, a named dataset (titanic, iris, penguins, tips), or a Kaggle slug.")
        sys.exit(1)

    console.print(Rule("[bold cyan]mlsherlock[/bold cyan]", style="cyan"))
    console.print(f"  Data source:  [bold]{data_source}[/bold]")
    console.print(f"  Target:       [bold]{target or '(agent will determine)'}[/bold]")
    console.print(f"  Task:         [bold]{task}[/bold]")
    console.print(f"  Output dir:   [bold]{output_dir}[/bold]")
    console.print(f"  Max iters:    [bold]{max_iterations}[/bold]")
    console.print(f"  Provider:     [bold]{provider}[/bold]")
    console.print(f"  Interactive:  [bold]{'no' if non_interactive else 'yes'}[/bold]")
    console.print(Rule(style="cyan"))

    from mlsherlock.ui.cli_callbacks import CliCallbacks
    from mlsherlock.engine.loop import AgentLoop
    from mlsherlock.engine.state import AgentState
    from mlsherlock.execution.sandbox import CodeExecutor

    state = AgentState(
        data_path=data_path,
        target_column=target or "",
        task=task,
        output_dir=output_dir,
        max_iterations=max_iterations,
    )
    callbacks = CliCallbacks(non_interactive=non_interactive, verbose=verbose)
    executor = CodeExecutor()

    loop = AgentLoop(state=state, callbacks=callbacks, executor=executor, provider=provider)

    try:
        loop.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        sys.exit(0)
    except Exception as exc:
        console.print(f"[bold red]Fatal error:[/bold red] {exc}")
        raise


if __name__ == "__main__":
    main()
