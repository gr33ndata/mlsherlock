"""Tool: save_plot — save the current matplotlib figure to output_dir."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlsherlock.engine.state import AgentState
    from mlsherlock.execution.sandbox import CodeExecutor
    from mlsherlock.engine.callbacks import BaseCallbacks


def run(
    filename: str,
    state: "AgentState",
    executor: "CodeExecutor",
    callbacks: "BaseCallbacks",
) -> str:
    """Save the current plt figure to output_dir/filename and return the path."""
    os.makedirs(state.output_dir, exist_ok=True)
    # Sanitise filename
    safe_name = os.path.basename(filename)
    if not safe_name.endswith((".png", ".jpg", ".pdf", ".svg")):
        safe_name += ".png"

    full_path = os.path.join(state.output_dir, safe_name)

    save_code = f"""
import os
os.makedirs({state.output_dir!r}, exist_ok=True)
plt.tight_layout()
plt.savefig({full_path!r}, dpi=120, bbox_inches='tight')
plt.clf()
print("saved")
"""
    output, error = executor.execute(save_code)
    if error:
        return f"[save_plot error] {error}"

    callbacks.on_plot(full_path)
    return f"Plot saved to: {full_path}"
