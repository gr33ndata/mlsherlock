"""CLI entry point: `mlsh train`."""
from __future__ import annotations

import os
import sys

import click
from dotenv import load_dotenv

load_dotenv()


@click.group()
def main() -> None:
    """mlsherlock — an intelligent ML agent that diagnoses and improves your models."""


@main.command()
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
    default="classification",
    show_default=True,
    type=click.Choice(["classification", "regression"], case_sensitive=False),
    help="ML task type.",
)
@click.option("--output-dir", default="./output", show_default=True, help="Directory for model and plots.")
@click.option("--max-iterations", default=20, show_default=True, type=int, help="Max agent iterations.")
@click.option(
    "--non-interactive",
    is_flag=True,
    default=False,
    help="Auto-approve all ask_user calls (picks first option).",
)
def train(
    data: str | None,
    target: str | None,
    task: str,
    output_dir: str,
    max_iterations: int,
    non_interactive: bool,
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
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[bold red]Error:[/bold red] ANTHROPIC_API_KEY environment variable not set.")
        console.print("Create a .env file with ANTHROPIC_API_KEY=your_key or export it in your shell.")
        sys.exit(1)

    # Resolve data source and initial message for the agent
    data_path = ""      # empty = agent will download it
    data_source = ""    # what to tell the agent

    _NAMED_DATASETS = {"titanic", "iris", "penguins", "diamonds", "tips"}

    if data is None:
        # No --data given: agent will ask interactively
        data_source = "(none — agent will ask)"
    elif os.path.exists(data):
        # Local file path
        if not target:
            console.print("[bold red]Error:[/bold red] --target is required when --data is a local file.")
            sys.exit(1)
        data_path = os.path.abspath(data)
        data_source = data_path
    elif data in _NAMED_DATASETS or ("/" in data and not data.startswith("/")):
        # Named dataset or Kaggle slug — agent will download it
        data_source = data
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
    console.print(f"  Interactive:  [bold]{'no' if non_interactive else 'yes'}[/bold]")
    console.print(Rule(style="cyan"))

    from mlsherlock.adapters.cli_callbacks import CliCallbacks
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
    callbacks = CliCallbacks(non_interactive=non_interactive)
    executor = CodeExecutor()

    loop = AgentLoop(state=state, callbacks=callbacks, executor=executor)

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
