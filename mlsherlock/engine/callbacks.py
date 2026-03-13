"""BaseCallbacks — the I/O interface between the engine and adapters."""
from abc import ABC, abstractmethod


class BaseCallbacks(ABC):
    @abstractmethod
    def on_message(self, text: str) -> None:
        """Called when the agent emits a text message."""

    @abstractmethod
    def on_tool_call(self, name: str, input_preview: str) -> None:
        """Called when a tool is about to be invoked."""

    @abstractmethod
    def on_tool_result(self, result: str, is_error: bool) -> None:
        """Called after a tool returns its result."""

    @abstractmethod
    def on_ask_user(self, question: str, options: list[str]) -> str:
        """Called when the agent needs human input. Must return the user's answer."""

    @abstractmethod
    def on_plot(self, path: str) -> None:
        """Called after a plot has been saved to *path*."""

    @abstractmethod
    def on_finish(self, summary: str, model_path: str) -> None:
        """Called when the agent calls `finish` with the final summary."""
