"""AgentState — shared mutable state for an ML session."""

import hashlib
from dataclasses import dataclass
from typing import Literal


@dataclass
class AgentState:
    # Session config
    data_path: str = ""
    target_column: str = ""
    task: Literal["classification", "regression"] = "classification"
    output_dir: str = "./output"
    max_iterations: int = 20

    # Loop control
    iteration: int = 0
    finished: bool = False

    # Error tracking for stuck-detection
    consecutive_errors: int = 0
    last_error_hash: str | None = None

    def record_error(self, error: str) -> None:
        h = hashlib.md5(error[:200].encode()).hexdigest()
        if h == self.last_error_hash:
            self.consecutive_errors += 1
        else:
            self.consecutive_errors = 1
            self.last_error_hash = h

    @property
    def is_stuck(self) -> bool:
        return self.consecutive_errors >= 3

    def reset_stuck_detection(self) -> None:
        self.consecutive_errors = 0
        self.last_error_hash = None
