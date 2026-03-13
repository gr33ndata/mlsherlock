"""Notebook adapter — future implementation using ipywidgets/IPython.display."""
from __future__ import annotations

from mlsherlock.engine.callbacks import BaseCallbacks


class NotebookCallbacks(BaseCallbacks):
    """Placeholder — not yet implemented."""

    def on_message(self, text: str) -> None:
        raise NotImplementedError("NotebookCallbacks not yet implemented")

    def on_tool_call(self, name: str, input_preview: str) -> None:
        raise NotImplementedError

    def on_tool_result(self, result: str, is_error: bool) -> None:
        raise NotImplementedError

    def on_ask_user(self, question: str, options: list[str]) -> str:
        raise NotImplementedError

    def on_plot(self, path: str) -> None:
        raise NotImplementedError

    def on_finish(self, summary: str, model_path: str) -> None:
        raise NotImplementedError
