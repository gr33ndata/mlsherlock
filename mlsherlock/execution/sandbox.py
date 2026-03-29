"""Shared execution sandbox with persistent globals and timeout support."""
import threading
from typing import Any

import numpy as np


def make_globals() -> dict[str, Any]:
    """Seed the shared namespace with common ML imports."""
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — must come before pyplot import
    import matplotlib.pyplot as plt
    import pandas as pd
    import sklearn

    g: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "sklearn": sklearn,
    }

    # Bring sklearn submodules in explicitly so the agent can reference them
    from sklearn import (
        datasets,
        ensemble,
        linear_model,
        metrics,
        model_selection,
        neighbors,
        pipeline,
        preprocessing,
        svm,
        tree,
    )
    g.update(
        {
            "datasets": datasets,
            "ensemble": ensemble,
            "linear_model": linear_model,
            "metrics": metrics,
            "model_selection": model_selection,
            "neighbors": neighbors,
            "pipeline": pipeline,
            "preprocessing": preprocessing,
            "svm": svm,
            "tree": tree,
        }
    )

    # Reproducibility: fixed seed
    np.random.seed(42)

    # plt.show() must be a no-op — figures are saved via save_plot tool
    plt.show = lambda: None

    return g


class CodeExecutor:
    """
    Executes Python code strings in a persistent shared namespace.

    State (variables, trained models, DataFrames) survives across calls.
    """

    def __init__(self) -> None:
        self._globals: dict[str, Any] = make_globals()

    @property
    def globals(self) -> dict[str, Any]:
        return self._globals

    def execute(self, code: str, timeout: float = 30.0) -> tuple[str, str]:
        """
        Run *code* in the shared namespace.

        Returns (stdout_output, error_message).  error_message is "" on success.
        Timeouts return a descriptive error string — they do NOT kill the thread
        (Python limitation).  Each execution runs in a shallow copy of globals so
        a timed-out thread cannot mutate the persistent sandbox state; the copy is
        merged back into self._globals only on successful completion.
        """
        from mlsherlock.execution.capture import ExecutionCapture

        # Shallow copy so a timed-out (still-running) thread writes into its own
        # dict and cannot corrupt the persistent globals.
        exec_globals: dict[str, Any] = dict(self._globals)
        result_holder: dict[str, Any] = {"capture": None, "error": ""}

        def run_code() -> None:
            cap = ExecutionCapture()
            with cap:
                try:
                    exec(code, exec_globals)  # noqa: S102
                except Exception as exc:  # noqa: BLE001
                    import traceback
                    result_holder["error"] = traceback.format_exc()
            result_holder["capture"] = cap

        thread = threading.Thread(target=run_code, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            # Thread keeps running in its own copy; self._globals is unaffected.
            return "", f"TimeoutError: execution exceeded {timeout}s"

        cap: ExecutionCapture = result_holder["capture"]
        if cap is None:
            return "", result_holder["error"]

        # Merge new/modified variables back into the persistent namespace.
        self._globals.update(exec_globals)
        return cap.combined, result_holder["error"]
