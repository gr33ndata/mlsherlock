"""AgentState — shared mutable state for an ML session."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentState:
    # Session config
    data_path: str = ""
    target_column: str = ""
    task: str = "classification"  # "classification" | "regression"
    output_dir: str = "./output"
    max_iterations: int = 20

    # Loop control
    iteration: int = 0
    finished: bool = False

    # Error tracking for stuck-detection
    _consecutive_same_error: int = 0
    _last_error_hash: Optional[str] = None

    def record_error(self, error: str) -> None:
        """Track repeated errors to detect the agent getting stuck."""
        import hashlib
        h = hashlib.md5(error[:200].encode()).hexdigest()
        if h == self._last_error_hash:
            self._consecutive_same_error += 1
        else:
            self._consecutive_same_error = 1
            self._last_error_hash = h

    @property
    def is_stuck(self) -> bool:
        """True if the same error has occurred 3+ times in a row."""
        return self._consecutive_same_error >= 3

    def reset_stuck_detection(self) -> None:
        """Reset the stuck-detection counters after injecting a hint."""
        self._consecutive_same_error = 0
        self._last_error_hash = None
