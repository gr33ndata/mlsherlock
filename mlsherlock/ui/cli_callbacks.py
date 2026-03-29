"""CLI adapter — WhatsApp-style chat output with Rich."""

import json
import os
import re
from datetime import datetime

from rich.console import Console
from rich.prompt import Prompt

console = Console()

AGENT_NAME  = "Sherlock"
AGENT_COLOR = "bold #C8A951"
USER_COLOR  = "bold #6BCB77"

RESULT_PREVIEW_LINES = 5
RESULT_PREVIEW_CHARS = 400


def ts() -> str:
    return datetime.now().strftime("%H:%M")


def first_sentence(text: str, maxlen: int = 120) -> str:
    """Extract the first meaningful sentence from agent text."""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.search(r"\.\s", line)
        if m and m.start() < maxlen:
            return line[: m.start() + 1]
        return line[:maxlen] + ("…" if len(line) > maxlen else "")
    return text[:maxlen]


def key_metric_line(result: str) -> str | None:
    """Return the first line that looks like a metric (contains % or accuracy/score words)."""
    for line in result.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.search(r"(%|accuracy|score|f1|auc|rmse|mae|r2)", line, re.IGNORECASE):
            return line[:120]
    return None


def compact_input(name: str, input_preview: str) -> str:
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


def trim_result(result: str) -> str:
    lines = [l for l in result.splitlines() if l.strip()]
    preview = "\n".join(lines[:RESULT_PREVIEW_LINES])
    if len(lines) > RESULT_PREVIEW_LINES:
        preview += f"\n… +{len(lines) - RESULT_PREVIEW_LINES} more lines"
    if len(preview) > RESULT_PREVIEW_CHARS:
        preview = preview[:RESULT_PREVIEW_CHARS] + "…"
    return preview


class CliCallbacks:
    def __init__(self, non_interactive: bool = False, verbose: bool = False) -> None:
        self._non_interactive = non_interactive
        self._verbose = verbose

    def on_message(self, text: str) -> None:
        if self._verbose:
            console.print(f"\n[{AGENT_COLOR}]{AGENT_NAME}[/{AGENT_COLOR}]  [dim]{ts()}[/dim]")
            for line in text.strip().splitlines():
                console.print(f"  {line}")
            console.print()
        else:
            summary = first_sentence(text)
            console.print(f"\n[{AGENT_COLOR}]{AGENT_NAME}[/{AGENT_COLOR}]  [dim]{ts()}[/dim]")
            console.print(f"  {summary}")

    def on_tool_call(self, name: str, input_preview: str) -> None:
        if self._verbose:
            label = compact_input(name, input_preview)
            console.print(f"  [dim]🔍 {name}  {label}[/dim]")
        # terse: silent — the result line tells the story

    def on_tool_result(self, result: str, is_error: bool) -> None:
        if is_error:
            first_line = next((l for l in result.splitlines() if l.strip()), result)
            console.print(f"  [red]✗ {first_line[:200]}[/red]")
        elif self._verbose:
            preview = trim_result(result)
            if preview and preview != "(no output)":
                for line in preview.splitlines():
                    console.print(f"  [dim]{line}[/dim]")
        else:
            metric = key_metric_line(result)
            if metric:
                console.print(f"  [dim]→ {metric}[/dim]")

    def on_ask_user(self, question: str, options: list[str]) -> str:
        if self._non_interactive:
            answer = options[0] if options else "yes"
            console.print(f"  [dim]auto → {answer}[/dim]")
            return answer

        console.print(f"\n[{AGENT_COLOR}]{AGENT_NAME}[/{AGENT_COLOR}]  [dim]{ts()}[/dim]")
        console.print(f"  {question}")
        console.print()
        if options:
            for i, opt in enumerate(options, 1):
                console.print(f"  [dim]{i}.[/dim] {opt}")
            console.print()
            raw = Prompt.ask(f"[{USER_COLOR}]You[/{USER_COLOR}]", default=options[0])
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(options):
                    return options[idx]
            except ValueError:
                pass
            return raw
        return Prompt.ask(f"[{USER_COLOR}]You[/{USER_COLOR}]")

    def on_plot(self, path: str) -> None:
        console.print(f"  [dim]📊 {os.path.basename(path)}[/dim]")

    def on_finish(self, summary: str, model_path: str) -> None:
        console.print(f"\n[{AGENT_COLOR}]{AGENT_NAME}[/{AGENT_COLOR}]  [dim]{ts()}[/dim]")
        console.print(f"  ✅ Done. Model saved to [bold]{model_path}[/bold]\n")
        for line in summary.strip().splitlines():
            console.print(f"  {line}")
        console.print()
