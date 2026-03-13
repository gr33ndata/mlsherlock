"""AgentLoop — manages conversation history, Claude API calls, and tool dispatch."""
from __future__ import annotations

import json
import os
from typing import Any

import anthropic
from dotenv import load_dotenv

from mlsherlock.engine.callbacks import BaseCallbacks
from mlsherlock.engine.state import AgentState
from mlsherlock.engine.system_prompt import get_system_prompt
from mlsherlock.execution.sandbox import CodeExecutor
from mlsherlock.tools.registry import dispatch, get_tool_schemas

load_dotenv()

_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 8096
# Rough token budget — trim history when we approach this
_CONTEXT_TRIM_THRESHOLD = 150_000


class AgentLoop:
    def __init__(
        self,
        state: AgentState,
        callbacks: BaseCallbacks,
        executor: CodeExecutor | None = None,
    ) -> None:
        self._state = state
        self._callbacks = callbacks
        self._executor = executor or CodeExecutor()
        self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self._history: list[dict[str, Any]] = []
        self._tools = get_tool_schemas()

    # ── Public ───────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Run the agent loop until `finish` is called or max_iterations reached."""
        # Seed the conversation with the task description
        self._history.append(
            {
                "role": "user",
                "content": self._initial_user_message(),
            }
        )

        while not self._state.finished and self._state.iteration < self._state.max_iterations:
            self._state.iteration += 1
            self._maybe_trim_history()
            self._maybe_inject_reminder()

            response = self._call_claude()
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
            # Agent needs to download the data — give it the source hint from data_path
            # or ask the user
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

    def _call_claude(self) -> anthropic.types.Message:
        return self._client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            system=get_system_prompt(),
            tools=self._tools,
            messages=self._history,
        )

    def _process_response(self, response: anthropic.types.Message) -> None:
        # Collect all content blocks for history
        assistant_content: list[dict[str, Any]] = []

        for block in response.content:
            if block.type == "text":
                self._callbacks.on_message(block.text)
                assistant_content.append({"type": "text", "text": block.text})

            elif block.type == "tool_use":
                # Notify callbacks
                input_preview = json.dumps(block.input, indent=2)[:500]
                self._callbacks.on_tool_call(block.name, input_preview)
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )

        # Append the full assistant turn
        self._history.append({"role": "assistant", "content": assistant_content})

        # Now dispatch tools sequentially and collect results
        tool_results: list[dict[str, Any]] = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            result = dispatch(
                block.name,
                block.input,
                self._state,
                self._executor,
                self._callbacks,
            )
            is_error = result.startswith("[") and "error" in result[:20].lower()
            self._callbacks.on_tool_result(result, is_error)

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                    "is_error": is_error,
                }
            )

            # If stuck, inject a hint as an extra tool result
            if self._state.is_stuck:
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": (
                            "[system hint] You have encountered the same error 3 times in a row. "
                            "Try a completely different approach or simplify the code."
                        ),
                        "is_error": False,
                    }
                )
                # Reset so hint isn't repeated every single call
                self._state._consecutive_same_error = 0

        if tool_results:
            self._history.append({"role": "user", "content": tool_results})

        # If stop_reason is end_turn with no tool calls, the agent is done talking
        if response.stop_reason == "end_turn" and not any(
            b.type == "tool_use" for b in response.content
        ):
            self._state.finished = True

    def _maybe_trim_history(self) -> None:
        """Drop oldest tool results when history grows large."""
        # Rough estimate: 4 chars ≈ 1 token
        approx_chars = sum(
            len(json.dumps(msg)) for msg in self._history
        )
        if approx_chars < _CONTEXT_TRIM_THRESHOLD * 4 * 0.8:
            return

        # Find and remove the oldest tool_result user turn
        for i, msg in enumerate(self._history):
            if msg["role"] == "user" and isinstance(msg["content"], list):
                if any(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in msg["content"]
                ):
                    self._history.pop(i)
                    return

    def _maybe_inject_reminder(self) -> None:
        """Inject an iteration progress reminder as a user message."""
        remaining = self._state.max_iterations - self._state.iteration
        if remaining in (5, 2):
            self._history.append(
                {
                    "role": "user",
                    "content": (
                        f"[system reminder] You have {remaining} iteration(s) remaining. "
                        "If you are satisfied with the current model, please call `finish` soon."
                    ),
                }
            )
