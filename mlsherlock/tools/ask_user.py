"""Tool: ask_user — delegate a question to the human via callbacks."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlsherlock.engine.callbacks import BaseCallbacks


def run(
    question: str,
    options: list[str],
    callbacks: "BaseCallbacks",
) -> str:
    """Ask the user a question and return their answer."""
    answer = callbacks.on_ask_user(question, options)
    return f"User answered: {answer}"
