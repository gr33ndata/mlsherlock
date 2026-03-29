"""Tool: ask_user — delegate a question to the human via callbacks."""
from __future__ import annotations

def run(
    question: str,
    options: list[str],
    callbacks,
) -> str:
    """Ask the user a question and return their answer."""
    answer = callbacks.on_ask_user(question, options)
    return f"User answered: {answer}"
