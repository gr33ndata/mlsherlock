"""Tool: finish — save the model, write a summary, and signal loop exit."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlsherlock.engine.state import AgentState
    from mlsherlock.execution.sandbox import CodeExecutor
    from mlsherlock.engine.callbacks import BaseCallbacks


def run(
    summary: str,
    model_variable: str,
    state: "AgentState",
    executor: "CodeExecutor",
    callbacks: "BaseCallbacks",
) -> str:
    """Persist the model and emit a final summary."""
    os.makedirs(state.output_dir, exist_ok=True)
    model_path = os.path.join(state.output_dir, "model.pkl")

    save_code = f"""
import joblib
import os
os.makedirs({state.output_dir!r}, exist_ok=True)
_model_obj = {model_variable}
joblib.dump(_model_obj, {model_path!r})
print(f"Model saved to {model_path!r}")
"""
    output, error = executor.execute(save_code)
    if error:
        return f"[finish error] Could not save model '{model_variable}': {error}"

    # Write summary file
    summary_path = os.path.join(state.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)

    state.finished = True
    callbacks.on_finish(summary, model_path)
    return f"Done. Model saved to {model_path}. Summary written to {summary_path}."
