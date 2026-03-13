"""Context manager that captures stdout/stderr during code execution."""
import io
import sys
from dataclasses import dataclass, field


@dataclass
class ExecutionCapture:
    """Captures stdout and stderr within a `with` block."""

    stdout: str = field(default="", init=False)
    stderr: str = field(default="", init=False)
    _old_stdout: object = field(default=None, init=False, repr=False)
    _old_stderr: object = field(default=None, init=False, repr=False)
    _buf_out: io.StringIO = field(default=None, init=False, repr=False)
    _buf_err: io.StringIO = field(default=None, init=False, repr=False)

    def __enter__(self) -> "ExecutionCapture":
        self._buf_out = io.StringIO()
        self._buf_err = io.StringIO()
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        sys.stdout = self._buf_out
        sys.stderr = self._buf_err
        return self

    def __exit__(self, *_) -> None:
        sys.stdout = self._old_stdout
        sys.stderr = self._old_stderr
        self.stdout = self._buf_out.getvalue()
        self.stderr = self._buf_err.getvalue()

    @property
    def combined(self) -> str:
        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(f"[stderr]\n{self.stderr}")
        return "\n".join(parts)
