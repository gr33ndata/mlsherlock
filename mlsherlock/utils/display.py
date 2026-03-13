"""Rich formatting helpers."""
from rich.console import Console
from rich.rule import Rule

console = Console()


def print_header(title: str) -> None:
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]", style="cyan"))


def print_error(message: str) -> None:
    console.print(f"[bold red]Error:[/bold red] {message}")
