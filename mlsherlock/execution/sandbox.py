"""Shared execution sandbox with persistent globals and timeout support."""
import threading
import traceback
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
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

from mlsherlock.execution.capture import ExecutionCapture


def make_globals() -> dict[str, Any]:
    """Seed the shared namespace with common ML imports."""
    g: dict[str, Any] = {
        "pd": pd, "np": np, "plt": plt, "sklearn": sklearn,
        "datasets": datasets, "ensemble": ensemble, "linear_model": linear_model,
        "metrics": metrics, "model_selection": model_selection, "neighbors": neighbors,
        "pipeline": pipeline, "preprocessing": preprocessing, "svm": svm, "tree": tree,
    }
    np.random.seed(42)
    plt.show = lambda: None
    return g


class CodeExecutor:
    """Executes Python code strings in a persistent shared namespace.

    State (variables, trained models, DataFrames) survives across calls.
    """

    def __init__(self) -> None:
        self._globals: dict[str, Any] = make_globals()

    @property
    def globals(self) -> dict[str, Any]:
        return self._globals

    def execute(self, code: str, timeout: float = 30.0) -> tuple[str, str]:
        """Run *code* in the shared namespace.

        Returns (stdout_output, error_message). error_message is "" on success.
        Each execution runs in a shallow copy of globals so a timed-out thread
        cannot mutate the persistent sandbox state; the copy is merged back only
        on successful completion.
        """
        exec_globals: dict[str, Any] = dict(self._globals)
        result_holder: dict[str, Any] = {"capture": None, "error": ""}

        def run_code() -> None:
            cap = ExecutionCapture()
            with cap:
                try:
                    exec(code, exec_globals)
                except Exception:
                    result_holder["error"] = traceback.format_exc()
            result_holder["capture"] = cap

        thread = threading.Thread(target=run_code, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            return "", f"TimeoutError: execution exceeded {timeout}s"

        cap: ExecutionCapture = result_holder["capture"]
        if cap is None:
            return "", result_holder["error"]

        self._globals.update(exec_globals)
        return cap.combined, result_holder["error"]
