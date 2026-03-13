"""Shared pytest fixtures."""
import os
import pytest
from mlsherlock.engine.callbacks import BaseCallbacks
from mlsherlock.engine.state import AgentState
from mlsherlock.execution.sandbox import CodeExecutor


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_CSV = os.path.join(FIXTURES_DIR, "sample.csv")


class FakeCallbacks(BaseCallbacks):
    """Captures all callback calls for inspection in tests."""

    def __init__(self) -> None:
        self.messages: list[str] = []
        self.tool_calls: list[tuple[str, str]] = []
        self.tool_results: list[tuple[str, bool]] = []
        self.plots: list[str] = []
        self.questions: list[tuple[str, list[str]]] = []
        self.finish_calls: list[tuple[str, str]] = []
        self._auto_answer: str = "yes"

    def on_message(self, text: str) -> None:
        self.messages.append(text)

    def on_tool_call(self, name: str, input_preview: str) -> None:
        self.tool_calls.append((name, input_preview))

    def on_tool_result(self, result: str, is_error: bool) -> None:
        self.tool_results.append((result, is_error))

    def on_ask_user(self, question: str, options: list[str]) -> str:
        self.questions.append((question, options))
        return options[0] if options else self._auto_answer

    def on_plot(self, path: str) -> None:
        self.plots.append(path)

    def on_finish(self, summary: str, model_path: str) -> None:
        self.finish_calls.append((summary, model_path))


@pytest.fixture
def executor():
    return CodeExecutor()


@pytest.fixture
def state(tmp_path):
    return AgentState(
        data_path=SAMPLE_CSV,
        target_column="target",
        task="classification",
        output_dir=str(tmp_path / "output"),
    )


@pytest.fixture
def callbacks():
    return FakeCallbacks()
