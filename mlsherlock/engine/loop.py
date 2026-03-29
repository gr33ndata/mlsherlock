"""AgentLoop — manages conversation history, provider API calls, and tool dispatch."""
from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv

from mlsherlock.engine.callbacks import BaseCallbacks
from mlsherlock.engine.providers import LLMProvider, NormalizedResponse, get_provider
from mlsherlock.engine.state import AgentState
from mlsherlock.engine.system_prompt import get_system_prompt
from mlsherlock.execution.sandbox import CodeExecutor
from mlsherlock.tools.registry import dispatch

load_dotenv()

# Rough token budget — trim history when we approach this
_CONTEXT_TRIM_THRESHOLD = 150_000


class AgentLoop:
    def __init__(
        self,
        state: AgentState,
        callbacks: BaseCallbacks,
        executor: CodeExecutor | None = None,
        provider: str = "openai",
    ) -> None:
        self._state = state
        self._callbacks = callbacks
        self._executor = executor or CodeExecutor()
        # Patch save_plot so inline calls from run_python use the correct output_dir
        _output_dir = state.output_dir
        def _save_plot(filename: str = "plot.png") -> str:
            import os
            import matplotlib.pyplot as plt
            os.makedirs(_output_dir, exist_ok=True)
            safe = os.path.basename(filename)
            if not safe.endswith((".png", ".jpg", ".pdf", ".svg")):
                safe += ".png"
            path = os.path.join(_output_dir, safe)
            plt.tight_layout()
            plt.savefig(path, dpi=120, bbox_inches="tight")
            plt.clf()
            return path
        self._executor.globals["save_plot"] = _save_plot
        self._provider: LLMProvider = get_provider(provider)
        self._history: list[dict[str, Any]] = []
        self._approx_history_chars: int = 0

    # ── Public ───────────────────────────────────────────────────────────────

    def _append_history(self, entry: dict[str, Any]) -> None:
        self._history.append(entry)
        self._approx_history_chars += len(json.dumps(entry))

    def run(self) -> None:
        """Run the agent loop until `finish` is called or max_iterations reached."""
        self._append_history({"role": "user", "content": self._initial_user_message()})

        while not self._state.finished and self._state.iteration < self._state.max_iterations:
            self._state.iteration += 1
            self._maybe_trim_history()
            self._maybe_inject_reminder()

            response = self._provider.call(self._history, get_system_prompt())
            self._process_response(response)

        if not self._state.finished:
            self._callbacks.on_message(
                f"Reached maximum iterations ({self._state.max_iterations}). "
                "Stopping without explicit finish."
            )

    # ── Private ──────────────────────────────────────────────────────────────

    def _initial_user_message(self) -> str:
        lines = [f"Please help me build a {self._state.task} model."]

        if self._state.data_path:
            lines.append(f"Dataset (local file): {self._state.data_path}")
            lines.append(f"Target column: {self._state.target_column}")
        else:
            lines.append(
                "No local dataset was provided. Please download an appropriate dataset using "
                "the `download_data` tool, then profile it and build the model."
            )
            if self._state.target_column:
                lines.append(f"Target column: {self._state.target_column}")

        lines += [
            f"Output directory: {self._state.output_dir}",
            "",
            "Start by reading and profiling the data, then train a baseline model, "
            "diagnose any issues, and iteratively improve. "
            "When you are satisfied with the results, call the `finish` tool.",
        ]
        return "\n".join(lines)

    def _process_response(self, response: NormalizedResponse) -> None:
        for text in response.text_blocks:
            self._callbacks.on_message(text)

        for tc in response.tool_calls:
            input_preview = json.dumps(tc.input, indent=2)[:500]
            self._callbacks.on_tool_call(tc.name, input_preview)

        self._append_history(self._provider.make_assistant_history_entry(response))

        tool_results: list[dict[str, Any]] = []
        inject_stuck_hint = False
        for tc in response.tool_calls:
            result = dispatch(tc.name, tc.input, self._state, self._executor, self._callbacks)
            is_error = result.startswith("[") and "error" in result[:20].lower()
            self._callbacks.on_tool_result(result, is_error)
            tool_results.append(
                {"tool_call_id": tc.id, "content": result, "is_error": is_error}
            )
            if self._state.is_stuck:
                inject_stuck_hint = True
                self._state.reset_stuck_detection()

        if tool_results:
            for entry in self._provider.make_tool_results_history_entries(tool_results):
                self._append_history(entry)

        # Inject stuck hint as a plain user message — avoids duplicate tool_call_ids
        if inject_stuck_hint:
            self._append_history({
                "role": "user",
                "content": (
                    "[system hint] You have encountered the same error 3 times in a row. "
                    "Try a completely different approach or simplify the code."
                ),
            })

        if response.is_done:
            self._state.finished = True

    def _maybe_trim_history(self) -> None:
        """Drop the oldest assistant+tool-results group when history grows large."""
        if self._approx_history_chars < _CONTEXT_TRIM_THRESHOLD * 4 * 0.8:
            return

        # Find the oldest assistant message that has tool calls, then remove it
        # together with all immediately following tool-result messages so the
        # history stays valid for both Anthropic and OpenAI.
        for i, msg in enumerate(self._history):
            if msg["role"] != "assistant":
                continue
            content = msg.get("content")
            has_tool_calls = (
                # OpenAI format
                bool(msg.get("tool_calls"))
                # Anthropic format
                or (
                    isinstance(content, list)
                    and any(
                        isinstance(b, dict) and b.get("type") == "tool_use"
                        for b in content
                    )
                )
            )
            if not has_tool_calls:
                continue
            # Found — collect the span: this assistant message + following tool results
            end = i + 1
            while end < len(self._history) and self._history[end]["role"] in ("tool", "user") and (
                self._history[end]["role"] == "tool"
                or (
                    isinstance(self._history[end].get("content"), list)
                    and any(
                        isinstance(b, dict) and b.get("type") == "tool_result"
                        for b in self._history[end]["content"]
                    )
                )
            ):
                end += 1
            self._approx_history_chars -= sum(
                len(json.dumps(m)) for m in self._history[i:end]
            )
            del self._history[i:end]
            return

    def _maybe_inject_reminder(self) -> None:
        remaining = self._state.max_iterations - self._state.iteration
        if remaining in (5, 2):
            self._append_history(
                {
                    "role": "user",
                    "content": (
                        f"[system reminder] You have {remaining} iteration(s) remaining. "
                        "If you are satisfied with the current model, please call `finish` soon."
                    ),
                }
            )
