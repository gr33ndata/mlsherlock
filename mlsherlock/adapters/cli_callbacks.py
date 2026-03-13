"""CLI adapter — implements BaseCallbacks with Rich terminal output."""
from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.text import Text

from mlsherlock.engine.callbacks import BaseCallbacks

console = Console()


class CliCallbacks(BaseCallbacks):
    def __init__(self, non_interactive: bool = False) -> None:
        self._non_interactive = non_interactive

    def on_message(self, text: str) -> None:
        console.print(Panel(text, title="[bold cyan]Agent[/bold cyan]", border_style="cyan"))

    def on_tool_call(self, name: str, input_preview: str) -> None:
        console.print(
            Panel(
                f"[bold yellow]{name}[/bold yellow]\n{input_preview}",
                title="[bold yellow]Tool Call[/bold yellow]",
                border_style="yellow",
            )
        )

    def on_tool_result(self, result: str, is_error: bool) -> None:
        style = "red" if is_error else "green"
        title = "[bold red]Error[/bold red]" if is_error else "[bold green]Result[/bold green]"
        # Truncate long results in display (full result still goes to agent)
        display = result if len(result) <= 2000 else result[:2000] + "\n... [truncated for display]"
        console.print(Panel(display, title=title, border_style=style))

    def on_ask_user(self, question: str, options: list[str]) -> str:
        if self._non_interactive:
            answer = options[0] if options else "yes"
            console.print(
                f"[dim][non-interactive] Auto-answering: {answer!r}[/dim]"
            )
            return answer

        console.print(
            Panel(question, title="[bold magenta]Agent Question[/bold magenta]", border_style="magenta")
        )
        if options:
            for i, opt in enumerate(options, 1):
                console.print(f"  [bold]{i}.[/bold] {opt}")
            raw = Prompt.ask("Your answer (enter text or option number)", default=options[0])
            # Accept numeric selection
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(options):
                    return options[idx]
            except ValueError:
                pass
            return raw
        else:
            return Prompt.ask("Your answer")

    def on_plot(self, path: str) -> None:
        console.print(f"[bold blue]Plot saved:[/bold blue] {path}")

    def on_finish(self, summary: str, model_path: str) -> None:
        console.print(
            Panel(
                f"[bold green]Model:[/bold green] {model_path}\n\n{summary}",
                title="[bold green]Session Complete[/bold green]",
                border_style="green",
            )
        )
