"""Tool: ask_user — delegate a question to the human via callbacks."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlsherlock.engine.state import AgentState
    from mlsherlock.execution.sandbox import CodeExecutor
    from mlsherlock.engine.callbacks import BaseCallbacks


def run(
    question: str,
    options: list[str],
    state: "AgentState",
    executor: "CodeExecutor",
    callbacks: "BaseCallbacks",
) -> str:
    """Ask the user a question and return their answer."""
    answer = callbacks.on_ask_user(question, options)
    return f"User answered: {answer}"
