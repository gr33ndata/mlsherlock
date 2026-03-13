"""CLI adapter — WhatsApp-style chat output with Rich."""
from __future__ import annotations

from datetime import datetime

from rich.console import Console
from rich.prompt import Prompt

from mlsherlock.engine.callbacks import BaseCallbacks

console = Console()

_AGENT_NAME  = "Sherlock"
_AGENT_COLOR = "bold #C8A951"   # warm amber — detective gold
_USER_COLOR  = "bold #6BCB77"   # WhatsApp-green for the user / system side

_RESULT_PREVIEW_LINES = 5
_RESULT_PREVIEW_CHARS = 400


def _ts() -> str:
    return datetime.now().strftime("%H:%M")


def _compact_input(name: str, input_preview: str) -> str:
    import json
    try:
        d = json.loads(input_preview)
    except Exception:
        return input_preview[:80]

    if name == "run_python":
        code = d.get("code", "")
        first = next((l.strip() for l in code.splitlines() if l.strip()), "")
        return first[:80] + ("…" if len(first) > 80 else "")
    if name == "read_data":
        return f"{d.get('path', '')}  target={d.get('target_column', '')}"
    if name in ("save_plot", "finish"):
        return d.get("filename") or d.get("summary", "")[:60]
    if name == "download_data":
        return d.get("source", "")
    return "  ".join(f"{k}={str(v)[:40]}" for k, v in d.items())


def _trim_result(result: str) -> str:
    lines = [l for l in result.splitlines() if l.strip()]
    preview = "\n".join(lines[:_RESULT_PREVIEW_LINES])
    if len(lines) > _RESULT_PREVIEW_LINES:
        preview += f"\n… +{len(lines) - _RESULT_PREVIEW_LINES} more lines"
    if len(preview) > _RESULT_PREVIEW_CHARS:
        preview = preview[:_RESULT_PREVIEW_CHARS] + "…"
    return preview


class CliCallbacks(BaseCallbacks):
    def __init__(self, non_interactive: bool = False) -> None:
        self._non_interactive = non_interactive

    def on_message(self, text: str) -> None:
        console.print(f"\n[{_AGENT_COLOR}]{_AGENT_NAME}[/{_AGENT_COLOR}]  [dim]{_ts()}[/dim]")
        for line in text.strip().splitlines():
            console.print(f"  {line}")
        console.print()

    def on_tool_call(self, name: str, input_preview: str) -> None:
        label = _compact_input(name, input_preview)
        console.print(f"  [dim]🔍 {name}  {label}[/dim]")

    def on_tool_result(self, result: str, is_error: bool) -> None:
        if is_error:
            first_line = next((l for l in result.splitlines() if l.strip()), result)
            console.print(f"  [red]✗ {first_line[:200]}[/red]")
        else:
            preview = _trim_result(result)
            if preview and preview != "(no output)":
                for line in preview.splitlines():
                    console.print(f"  [dim]{line}[/dim]")

    def on_ask_user(self, question: str, options: list[str]) -> str:
        if self._non_interactive:
            answer = options[0] if options else "yes"
            console.print(f"  [dim]auto → {answer}[/dim]")
            return answer

        console.print(f"\n[{_AGENT_COLOR}]{_AGENT_NAME}[/{_AGENT_COLOR}]  [dim]{_ts()}[/dim]")
        console.print(f"  {question}")
        console.print()
        if options:
            for i, opt in enumerate(options, 1):
                console.print(f"  [dim]{i}.[/dim] {opt}")
            console.print()
            raw = Prompt.ask(f"[{_USER_COLOR}]You[/{_USER_COLOR}]", default=options[0])
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(options):
                    return options[idx]
            except ValueError:
                pass
            return raw
        return Prompt.ask(f"[{_USER_COLOR}]You[/{_USER_COLOR}]")

    def on_plot(self, path: str) -> None:
        import os
        console.print(f"  [dim]📊 saved → {os.path.basename(path)}[/dim]")

    def on_finish(self, summary: str, model_path: str) -> None:
        console.print(f"\n[{_AGENT_COLOR}]{_AGENT_NAME}[/{_AGENT_COLOR}]  [dim]{_ts()}[/dim]")
        console.print(f"  ✅ Done. Model saved to [bold]{model_path}[/bold]\n")
        for line in summary.strip().splitlines():
            console.print(f"  {line}")
        console.print()
