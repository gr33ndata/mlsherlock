"""LLM provider abstraction — Anthropic and OpenAI."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from mlsherlock.tools.registry import get_tool_schemas


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class NormalizedResponse:
    text_blocks: list[str]
    tool_calls: list[ToolCall]
    is_done: bool


class AnthropicProvider:
    _MODEL = "claude-sonnet-4-6"
    _MAX_TOKENS = 8096

    def __init__(self) -> None:
        import anthropic
        self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self._schemas = get_tool_schemas()

    def call(self, history: list[dict], system_prompt: str) -> NormalizedResponse:
        response = self._client.messages.create(
            model=self._MODEL,
            max_tokens=self._MAX_TOKENS,
            system=system_prompt,
            tools=self._schemas,
            messages=history,
        )
        text_blocks = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text_blocks.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=block.input))

        is_done = response.stop_reason == "end_turn" and not tool_calls
        return NormalizedResponse(text_blocks=text_blocks, tool_calls=tool_calls, is_done=is_done)

    def make_assistant_history_entry(self, response: NormalizedResponse) -> dict[str, Any]:
        content: list[dict] = []
        for text in response.text_blocks:
            content.append({"type": "text", "text": text})
        for tc in response.tool_calls:
            content.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input})
        return {"role": "assistant", "content": content}

    def make_tool_results_history_entries(self, tool_results: list[dict]) -> list[dict[str, Any]]:
        return [{"role": "user", "content": [
            {
                "type": "tool_result",
                "tool_use_id": r["tool_call_id"],
                "content": r["content"],
                "is_error": r["is_error"],
            }
            for r in tool_results
        ]}]


class OpenAIProvider:
    _MODEL = "gpt-4o"
    _MAX_TOKENS = 8096

    def __init__(self) -> None:
        import openai
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
        org = os.environ.get("OPENAI_ORG_ID") or os.environ.get("OPENAI_ORG")
        self._client = openai.OpenAI(api_key=api_key, organization=org or None)
        self._schemas = self._convert_schemas(get_tool_schemas())

    def _convert_schemas(self, anthropic_schemas: list[dict]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s["description"],
                    "parameters": s["input_schema"],
                },
            }
            for s in anthropic_schemas
        ]

    def call(self, history: list[dict], system_prompt: str) -> NormalizedResponse:
        messages = [{"role": "system", "content": system_prompt}] + history
        response = self._client.chat.completions.create(
            model=self._MODEL,
            max_tokens=self._MAX_TOKENS,
            tools=self._schemas,
            messages=messages,
        )
        message = response.choices[0].message
        text_blocks = [message.content] if message.content else []
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    input=json.loads(tc.function.arguments),
                ))

        is_done = response.choices[0].finish_reason == "stop" and not tool_calls
        return NormalizedResponse(text_blocks=text_blocks, tool_calls=tool_calls, is_done=is_done)

    def make_assistant_history_entry(self, response: NormalizedResponse) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "role": "assistant",
            "content": " ".join(response.text_blocks) if response.text_blocks else None,
        }
        if response.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.input)},
                }
                for tc in response.tool_calls
            ]
        return entry

    def make_tool_results_history_entries(self, tool_results: list[dict]) -> list[dict[str, Any]]:
        return [
            {"role": "tool", "tool_call_id": r["tool_call_id"], "content": r["content"]}
            for r in tool_results
        ]


def get_provider(name: str) -> AnthropicProvider | OpenAIProvider:
    if name == "anthropic":
        return AnthropicProvider()
    if name == "openai":
        return OpenAIProvider()
    raise ValueError(f"Unknown provider: {name!r}. Choose 'anthropic' or 'openai'.")
