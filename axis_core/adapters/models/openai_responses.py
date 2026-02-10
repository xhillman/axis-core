"""OpenAI Responses API adapter implementation."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any, cast

try:
    from openai import AsyncOpenAI
except ImportError as e:
    raise ImportError(
        "OpenAIResponsesModel requires the openai package. "
        "Install with: pip install axis-core[openai]"
    ) from e

from axis_core.adapters.models.openai import MODEL_PRICING
from axis_core.errors import ModelError
from axis_core.protocols.model import ModelChunk, ModelResponse, ToolCall, UsageStats
from axis_core.tool import ToolManifest


class OpenAIResponsesModel:
    """OpenAI adapter backed by the Responses API."""

    def __init__(
        self,
        model_id: str,
        api_key: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
    ) -> None:
        self._model_id = model_id
        self._temperature = temperature
        self._max_tokens = max_tokens

        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Provide it via api_key parameter or set OPENAI_API_KEY environment variable."
            )

        self._client = AsyncOpenAI(api_key=self._api_key)

    @property
    def model_id(self) -> str:
        """Return the model identifier."""
        return self._model_id

    @staticmethod
    def _normalize_message_content(content: Any) -> str:
        """Normalize message content to plain text for Responses input."""
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    if "text" in item:
                        parts.append(str(item["text"]))
                    elif "content" in item:
                        parts.append(str(item["content"]))
                    else:
                        parts.append(json.dumps(item))
                    continue
                parts.append(str(item))
            return " ".join(parts)

        if content is None:
            return ""

        if isinstance(content, dict):
            if "text" in content:
                return str(content["text"])
            return json.dumps(content)

        return str(content)

    @staticmethod
    def _convert_tool_manifest_to_responses(manifest: ToolManifest) -> dict[str, Any]:
        """Convert ToolManifest into OpenAI Responses function tool format."""
        return {
            "type": "function",
            "name": manifest.name,
            "description": manifest.description,
            "parameters": manifest.input_schema,
        }

    @staticmethod
    def _convert_tools_to_openai(tools: Any) -> list[dict[str, Any]] | None:
        """Convert tools input to OpenAI Responses tools list."""
        if tools is None:
            return None
        if not tools:
            return None

        if isinstance(tools, list) and tools:
            first_tool = tools[0]
            if isinstance(first_tool, dict):
                tool_dicts: list[dict[str, Any]] = tools
                return tool_dicts
            if isinstance(first_tool, ToolManifest):
                return [
                    OpenAIResponsesModel._convert_tool_manifest_to_responses(tool)
                    for tool in tools
                ]
        return None

    @staticmethod
    def _convert_messages_to_responses_input(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert internal chat-style messages to Responses input items."""
        converted: list[dict[str, Any]] = []

        for msg in messages:
            role = str(msg.get("role", "user"))
            content = OpenAIResponsesModel._normalize_message_content(msg.get("content"))

            if role == "tool":
                call_id = str(msg.get("tool_call_id", "")).strip()
                if not call_id:
                    call_id = f"tool_call_{len(converted)}"
                converted.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": content,
                })
                continue

            if role in {"user", "assistant", "system", "developer"} and content:
                converted.append({
                    "type": "message",
                    "role": role,
                    "content": content,
                })

            tool_calls = msg.get("tool_calls")
            if role == "assistant" and isinstance(tool_calls, list):
                for idx, tc in enumerate(tool_calls):
                    if not isinstance(tc, dict):
                        continue
                    name = tc.get("name")
                    if not isinstance(name, str) or not name:
                        continue
                    arguments = tc.get("arguments", {})
                    if isinstance(arguments, dict):
                        arguments_text = json.dumps(arguments)
                    else:
                        arguments_text = str(arguments) if arguments is not None else "{}"

                    raw_call_id = tc.get("id")
                    if isinstance(raw_call_id, str) and raw_call_id:
                        call_id = raw_call_id
                    else:
                        call_id = f"call_{len(converted)}_{idx}"

                    converted.append({
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": arguments_text,
                    })

        return converted

    @staticmethod
    def _apply_stop_sequences(content: str, stop_sequences: list[str] | None) -> str:
        """Trim content at the earliest stop sequence if provided."""
        if not stop_sequences:
            return content

        end_index = len(content)
        for stop in stop_sequences:
            if not stop:
                continue
            index = content.find(stop)
            if index != -1 and index < end_index:
                end_index = index

        return content[:end_index]

    @staticmethod
    def _extract_tool_calls(response: Any) -> tuple[ToolCall, ...] | None:
        """Extract function tool calls from a Responses API payload."""
        output = getattr(response, "output", None)
        if not output:
            return None

        calls: list[ToolCall] = []
        for idx, item in enumerate(output):
            if getattr(item, "type", None) != "function_call":
                continue

            name = getattr(item, "name", None)
            if not isinstance(name, str) or not name:
                continue

            raw_args = getattr(item, "arguments", "{}")
            if isinstance(raw_args, dict):
                arguments = raw_args
            else:
                try:
                    arguments = json.loads(str(raw_args))
                except json.JSONDecodeError:
                    arguments = {}

            call_id = getattr(item, "call_id", None) or getattr(item, "id", None) or f"call_{idx}"
            calls.append(
                ToolCall(
                    id=str(call_id),
                    name=name,
                    arguments=arguments,
                )
            )

        return tuple(calls) if calls else None

    @staticmethod
    def _usage_from_response(response: Any) -> UsageStats:
        """Map Responses usage fields into canonical UsageStats."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return UsageStats.from_openai(
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )

        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", input_tokens + output_tokens) or 0)

        return UsageStats.from_openai(
            {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": total_tokens,
            }
        )

    async def complete(
        self,
        messages: Any,
        system: str | None = None,
        tools: Any | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ModelResponse:
        """Complete a prompt using the OpenAI Responses API."""
        try:
            input_items = self._convert_messages_to_responses_input(messages)
            kwargs: dict[str, Any] = {
                "model": self._model_id,
                "input": input_items,
                "temperature": temperature if temperature is not None else self._temperature,
            }

            if system is not None:
                kwargs["instructions"] = system

            tokens_value = max_tokens if max_tokens is not None else self._max_tokens
            if tokens_value is not None:
                kwargs["max_output_tokens"] = tokens_value

            openai_tools = self._convert_tools_to_openai(tools)
            if openai_tools is not None:
                kwargs["tools"] = openai_tools

            if metadata is not None:
                kwargs["metadata"] = metadata

            response = await self._client.responses.create(**kwargs)

            content = cast(str, getattr(response, "output_text", "") or "")
            content = self._apply_stop_sequences(content, stop_sequences)
            tool_calls = self._extract_tool_calls(response)
            usage = self._usage_from_response(response)
            cost = self.estimate_cost(usage.input_tokens, usage.output_tokens)

            return ModelResponse(
                content=content,
                tool_calls=tool_calls,
                usage=usage,
                cost_usd=cost,
            )
        except Exception as e:
            raise self._classify_openai_error(e) from e

    async def stream(
        self,
        messages: Any,
        system: str | None = None,
        tools: Any | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[ModelChunk]:
        """Stream a completion using the OpenAI Responses API."""
        del stop_sequences  # Responses streaming does not support server-side stop sequences.
        try:
            input_items = self._convert_messages_to_responses_input(messages)
            kwargs: dict[str, Any] = {
                "model": self._model_id,
                "input": input_items,
                "temperature": temperature if temperature is not None else self._temperature,
                "stream": True,
            }

            if system is not None:
                kwargs["instructions"] = system

            tokens_value = max_tokens if max_tokens is not None else self._max_tokens
            if tokens_value is not None:
                kwargs["max_output_tokens"] = tokens_value

            openai_tools = self._convert_tools_to_openai(tools)
            if openai_tools is not None:
                kwargs["tools"] = openai_tools

            if metadata is not None:
                kwargs["metadata"] = metadata

            stream = cast(AsyncIterator[Any], self._client.responses.create(**kwargs))

            async for event in stream:
                event_type = getattr(event, "type", "")

                if event_type == "response.output_text.delta":
                    yield ModelChunk(
                        content=str(getattr(event, "delta", "")),
                        tool_call_delta=None,
                        is_final=False,
                    )
                    continue

                if event_type == "response.function_call_arguments.delta":
                    yield ModelChunk(
                        content="",
                        tool_call_delta={"partial_json": str(getattr(event, "delta", ""))},
                        is_final=False,
                    )
                    continue

                if event_type == "response.completed":
                    yield ModelChunk(content="", tool_call_delta=None, is_final=True)
                    break

                if event_type in {"response.failed", "response.error"}:
                    error_message = f"Responses stream failed for {self._model_id}"
                    details = {"error_type": "api_error", "event_type": event_type}
                    raise ModelError(
                        message=error_message,
                        model_id=self._model_id,
                        recoverable=False,
                        details=details,
                    )
        except Exception as e:
            if isinstance(e, ModelError):
                raise
            raise self._classify_openai_error(e) from e

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD for token usage."""
        pricing = MODEL_PRICING.get(self._model_id, {})
        if not pricing:
            return 0.0

        input_cost = (input_tokens / 1_000_000) * pricing.get("input_per_mtok", 0.0)
        output_cost = (output_tokens / 1_000_000) * pricing.get("output_per_mtok", 0.0)
        return input_cost + output_cost

    def _classify_openai_error(self, error: Exception) -> ModelError:
        """Normalize OpenAI SDK errors into ModelError."""
        import openai

        if isinstance(error, openai.RateLimitError):
            return ModelError(
                message=f"Rate limit exceeded for {self._model_id}: {error}",
                model_id=self._model_id,
                recoverable=True,
                details={"error_type": "rate_limit"},
                cause=error,
            )

        if isinstance(error, openai.AuthenticationError):
            return ModelError(
                message=f"Authentication failed for {self._model_id}: {error}",
                model_id=self._model_id,
                recoverable=False,
                details={"error_type": "authentication"},
                cause=error,
            )

        if isinstance(error, openai.BadRequestError):
            return ModelError(
                message=f"Bad request to {self._model_id}: {error}",
                model_id=self._model_id,
                recoverable=False,
                details={"error_type": "bad_request"},
                cause=error,
            )

        if isinstance(error, openai.APITimeoutError):
            return ModelError(
                message=f"API timeout for {self._model_id}: {error}",
                model_id=self._model_id,
                recoverable=True,
                details={"error_type": "timeout"},
                cause=error,
            )

        if isinstance(error, openai.APIError):
            return ModelError(
                message=f"API error for {self._model_id}: {error}",
                model_id=self._model_id,
                recoverable=False,
                details={"error_type": "api_error"},
                cause=error,
            )

        return ModelError(
            message=f"Unexpected error for {self._model_id}: {error}",
            model_id=self._model_id,
            recoverable=False,
            details={"error_type": "unknown"},
            cause=error,
        )
