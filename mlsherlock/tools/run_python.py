"""Tool: run_python — execute arbitrary Python in the sandbox."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlsherlock.engine.state import AgentState
    from mlsherlock.execution.sandbox import CodeExecutor
    from mlsherlock.engine.callbacks import BaseCallbacks

_MAX_OUTPUT_CHARS = 8_000
_DF_PREVIEW_ROWS = 20


def run(
    code: str,
    state: "AgentState",
    executor: "CodeExecutor",
    callbacks: "BaseCallbacks",
) -> str:
    """Execute *code* and return a text result for the agent."""
    output, error = executor.execute(code)

    # Truncate large outputs to protect context window
    combined = (output or "") + (f"\n[error]\n{error}" if error else "")
    if len(combined) > _MAX_OUTPUT_CHARS:
        combined = combined[:_MAX_OUTPUT_CHARS] + f"\n... [truncated at {_MAX_OUTPUT_CHARS} chars]"

    if error:
        state.record_error(error)
        return f"[execution error]\n{combined}"

    return combined or "(no output)"
