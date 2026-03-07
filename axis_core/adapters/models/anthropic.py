"""Anthropic model adapter implementation.

This module provides the AnthropicModel adapter for Claude models via the Anthropic API.
Requires the 'anthropic' package: pip install axis-core[anthropic]
"""

import os
from collections.abc import AsyncIterator
from typing import Any

# Conditional import per AD-040
try:
    import anthropic
    from anthropic import AsyncAnthropic
except ImportError as e:
    raise ImportError(
        "AnthropicModel requires the anthropic package. "
        "Install with: pip install axis-core[anthropic]"
    ) from e

from axis_core.adapters.models.provider_helpers import (
    DEFAULT_SCHEMA_FIELDS_TO_STRIP,
    DEFAULT_TOOL_CALL_ID_INVALID_CHARS,
    normalize_tool_call_id,
    sanitize_tool_schema_for_provider,
)
from axis_core.errors import ModelError
from axis_core.protocols.model import ModelChunk, ModelResponse, NormalizedUsage, ToolCall
from axis_core.tool import ToolManifest

# Pricing table for cost estimation (per million tokens)
# Source: https://docs.anthropic.com/en/docs/about-claude/models/all-models (as of 2026-02)
MODEL_PRICING: dict[str, dict[str, float]] = {
  # Exact API model names
  "claude-opus-4-20250514": {
    "input_per_mtok": 15.00,
    "output_per_mtok": 75.00
  },
  "claude-opus-4-1-20250805": {
    "input_per_mtok": 15.00,
    "output_per_mtok": 75.00
  },
  "claude-opus-4-5-20251101": {
    "input_per_mtok": 5.00,
    "output_per_mtok": 25.00
  },
  "claude-opus-4-6": {
    "input_per_mtok": 5.00,
    "output_per_mtok": 25.00
  },
  "claude-sonnet-4-20250514": {
    "input_per_mtok": 3.00,
    "output_per_mtok": 15.00
  },
  "claude-sonnet-4-5-20250929": {
    "input_per_mtok": 3.00,
    "output_per_mtok": 15.00
  },
  "claude-haiku-4-5-20251001": {
    "input_per_mtok": 1.00,
    "output_per_mtok": 5.00
  },
  "claude-3-haiku-20240307": {
    "input_per_mtok": 0.25,
    "output_per_mtok": 1.25
  },
  # Convenience aliases (point to latest versions)
  "claude-opus": {
    "input_per_mtok": 5.00,
    "output_per_mtok": 25.00
  },
  "claude-sonnet": {
    "input_per_mtok": 3.00,
    "output_per_mtok": 15.00
  },
  "claude-haiku": {
    "input_per_mtok": 1.00,
    "output_per_mtok": 5.00
  }
}


class AnthropicModel:
    """Anthropic Claude model adapter.

    Provides access to Claude models through the Anthropic Messages API.
    Supports streaming, tool use, and cost tracking.

    Args:
        model_id: Model identifier (e.g., 'claude-sonnet-4-20250514')
        api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        temperature: Sampling temperature (0.0-2.0, default 1.0)
        max_tokens: Maximum tokens to generate (default 4096)

    Example:
        >>> model = AnthropicModel(model_id="claude-sonnet-4-20250514")
        >>> response = await model.complete(
        ...     messages=[{"role": "user", "content": "Hello"}],
        ...     system="You are a helpful assistant."
        ... )
        >>> print(response.content)
        Hello! How can I help you today?
    """

    _TOOL_CALL_ID_MAX_LEN = 64
    _TOOL_CALL_ID_PREFIX = "toolu"
    _TOOL_CALL_ID_INVALID_CHARS = DEFAULT_TOOL_CALL_ID_INVALID_CHARS
    _SCHEMA_FIELDS_TO_STRIP = DEFAULT_SCHEMA_FIELDS_TO_STRIP

    def __init__(
        self,
        model_id: str,
        api_key: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
    ) -> None:
        """Initialize the Anthropic model adapter.

        Args:
            model_id: Model identifier
            api_key: API key (defaults to ANTHROPIC_API_KEY env var)
            temperature: Default temperature for completions
            max_tokens: Default max tokens for completions

        Raises:
            ValueError: If api_key is not provided and ANTHROPIC_API_KEY is not set
        """
        self._model_id = model_id
        self._temperature = temperature
        self._max_tokens = max_tokens

        # Get API key from parameter or environment
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Anthropic API key is required. "
                "Provide it via api_key parameter or set ANTHROPIC_API_KEY environment variable."
            )

        # Initialize Anthropic client
        self._client = AsyncAnthropic(api_key=self._api_key)

    @property
    def model_id(self) -> str:
        """Return the model identifier."""
        return self._model_id

    @classmethod
    def _map_anthropic_error(
        cls,
        error: Exception,
    ) -> tuple[str, bool, int | None, str | None]:
        """Normalize Anthropic SDK exceptions for fallback semantics."""
        api_connection_error = getattr(anthropic, "APIConnectionError", None)
        permission_denied_error = getattr(anthropic, "PermissionDeniedError", None)
        auth_types = (anthropic.AuthenticationError,) + (
            (permission_denied_error,) if isinstance(permission_denied_error, type) else ()
        )
        bad_request_types = (anthropic.BadRequestError,)

        if isinstance(error, anthropic.RateLimitError):
            explicit_reason = "rate_limit"
            provider_error_reason = None
        elif isinstance(error, anthropic.APITimeoutError):
            explicit_reason = "timeout"
            provider_error_reason = None
        elif api_connection_error is not None and isinstance(error, api_connection_error):
            explicit_reason = "connection_error"
            provider_error_reason = None
        elif isinstance(error, auth_types):
            explicit_reason = "authentication"
            provider_error_reason = None
        elif isinstance(error, bad_request_types):
            explicit_reason = "invalid_request"
            provider_error_reason = None
        elif isinstance(error, anthropic.APIError):
            explicit_reason = None
            provider_error_reason = "provider_error"
        else:
            explicit_reason = None
            provider_error_reason = None

        normalized = ModelError.normalize_provider_error(
            error,
            explicit_reason=explicit_reason,
            provider_error_reason=provider_error_reason,
        )

        return (
            normalized.reason,
            normalized.recoverable,
            normalized.status_code,
            normalized.provider_code,
        )

    @staticmethod
    def _convert_tool_manifest_to_anthropic(manifest: ToolManifest) -> dict[str, Any]:
        """Convert a ToolManifest to Anthropic's tool format.

        This adapter method transforms the protocol-defined ToolManifest
        into the format expected by Anthropic's API.

        Args:
            manifest: Tool manifest from the protocol layer

        Returns:
            Dict in Anthropic's tool format with name, description, and input_schema

        Example:
            >>> manifest = ToolManifest(
            ...     name="get_weather",
            ...     description="Get weather",
            ...     input_schema={"type": "object", "properties": {...}},
            ...     ...
            ... )
            >>> schema = AnthropicModel._convert_tool_manifest_to_anthropic(manifest)
            >>> schema["name"]
            "get_weather"
        """
        input_schema = AnthropicModel._sanitize_tool_schema_for_provider(manifest.input_schema)
        return {
            "name": manifest.name,
            "description": manifest.description,
            "input_schema": input_schema,
        }

    @classmethod
    def _sanitize_tool_schema_for_provider(cls, schema: Any) -> dict[str, Any]:
        """Sanitize schema for Anthropic tool compatibility."""
        return sanitize_tool_schema_for_provider(
            schema,
            fields_to_strip=cls._SCHEMA_FIELDS_TO_STRIP,
        )

    @classmethod
    def _normalize_tool_call_id(
        cls,
        raw_id: Any,
        *,
        next_index: int,
        id_map: dict[str, str],
        used_ids: set[str],
    ) -> str:
        """Normalize tool call IDs to provider-safe format and maintain stable mapping."""
        return normalize_tool_call_id(
            raw_id,
            next_index=next_index,
            id_map=id_map,
            used_ids=used_ids,
            invalid_chars=cls._TOOL_CALL_ID_INVALID_CHARS,
            prefix=cls._TOOL_CALL_ID_PREFIX,
            max_len=cls._TOOL_CALL_ID_MAX_LEN,
        )

    @staticmethod
    def _convert_tools_to_anthropic(tools: Any) -> list[dict[str, Any]] | None:
        """Convert tools parameter to Anthropic format.

        Handles both ToolManifest objects (protocol layer) and raw dicts
        (for backward compatibility or direct API usage).

        Args:
            tools: List of ToolManifest objects, list of dicts, or None

        Returns:
            List of tool dicts in Anthropic format, or None if no tools

        Example:
            >>> manifests = [ToolManifest(...), ToolManifest(...)]
            >>> anthropic_tools = AnthropicModel._convert_tools_to_anthropic(manifests)
            >>> len(anthropic_tools)
            2
        """
        if tools is None:
            return None

        if not tools:
            return None

        # Check if tools are ToolManifest objects
        if isinstance(tools, list) and tools:
            first_tool = tools[0]

            # If already dicts, pass through (backward compatibility)
            if isinstance(first_tool, dict):
                # Type narrowing: we know it's a list of dicts
                tool_dicts: list[dict[str, Any]] = tools
                return tool_dicts

            # If ToolManifest objects, convert them
            if isinstance(first_tool, ToolManifest):
                return [
                    AnthropicModel._convert_tool_manifest_to_anthropic(manifest)
                    for manifest in tools
                ]

        # Shouldn't reach here with proper input types
        return None

    @classmethod
    def _convert_messages_to_anthropic(cls, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert messages from internal format to Anthropic's API format.

        The internal format uses OpenAI-style conventions:
        - Assistant messages may have a 'tool_calls' field
        - Tool results use role='tool' with 'tool_call_id'

        Anthropic's format differs:
        - Assistant messages use content blocks with type='tool_use'
        - Tool results use role='user' with content blocks of type='tool_result'

        Args:
            messages: List of messages in internal format

        Returns:
            List of messages in Anthropic's API format

        Example:
            Internal format:
                [
                    {"role": "user", "content": "What's the weather?"},
                    {"role": "assistant", "content": "Let me check.",
                     "tool_calls": [{"id": "tc1", "name": "get_weather", "arguments": {...}}]},
                    {"role": "tool", "tool_call_id": "tc1", "content": "Sunny, 72°F"}
                ]

            Anthropic format:
                [
                    {"role": "user", "content": "What's the weather?"},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": "Let me check."},
                        {"type": "tool_use", "id": "tc1", "name": "get_weather", "input": {...}}
                    ]},
                    {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "tc1", "content": "Sunny, 72°F"}
                    ]}
                ]
        """
        converted: list[dict[str, Any]] = []
        pending_tool_results: list[dict[str, Any]] = []
        tool_call_id_map: dict[str, str] = {}
        used_tool_call_ids: set[str] = set()
        generated_id_index = 1

        for msg in messages:
            role = msg.get("role", "")

            if role == "tool":
                normalized_id = cls._normalize_tool_call_id(
                    msg.get("tool_call_id"),
                    next_index=generated_id_index,
                    id_map=tool_call_id_map,
                    used_ids=used_tool_call_ids,
                )
                generated_id_index += 1

                # Collect tool results - they'll be batched into a single user message
                tool_result_block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": normalized_id,
                    "content": msg.get("content", ""),
                }
                pending_tool_results.append(tool_result_block)

            elif role == "assistant":
                # Flush any pending tool results first (shouldn't happen, but be safe)
                if pending_tool_results:
                    converted.append({
                        "role": "user",
                        "content": pending_tool_results,
                    })
                    pending_tool_results = []

                # Convert assistant message
                tool_calls = msg.get("tool_calls", [])
                content = msg.get("content", "")

                if tool_calls:
                    # Build content array with text and tool_use blocks
                    content_blocks: list[dict[str, Any]] = []

                    # Add text block if there's text content
                    if content:
                        content_blocks.append({"type": "text", "text": content})

                    # Add tool_use blocks
                    for tc in tool_calls:
                        if not isinstance(tc, dict):
                            continue
                        normalized_id = cls._normalize_tool_call_id(
                            tc.get("id"),
                            next_index=generated_id_index,
                            id_map=tool_call_id_map,
                            used_ids=used_tool_call_ids,
                        )
                        generated_id_index += 1
                        content_blocks.append({
                            "type": "tool_use",
                            "id": normalized_id,
                            "name": tc.get("name", ""),
                            "input": tc.get("arguments", {}),
                        })

                    converted.append({
                        "role": "assistant",
                        "content": content_blocks,
                    })
                else:
                    # Simple text-only assistant message
                    converted.append({
                        "role": "assistant",
                        "content": content,
                    })

            elif role == "user":
                # Flush pending tool results before user message
                if pending_tool_results:
                    converted.append({
                        "role": "user",
                        "content": pending_tool_results,
                    })
                    pending_tool_results = []

                # Pass through user messages as-is
                converted.append(msg)

            else:
                # Unknown role - pass through (shouldn't happen)
                converted.append(msg)

        # Flush any remaining tool results at the end
        if pending_tool_results:
            converted.append({
                "role": "user",
                "content": pending_tool_results,
            })

        return converted

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
        """Complete a prompt with the model (non-streaming).

        Args:
            messages: List of message dicts with 'role' and 'content'
            system: System prompt/instructions
            tools: Available tools (ToolManifest objects or dicts)
            temperature: Sampling temperature (overrides default)
            max_tokens: Maximum tokens to generate (overrides default)
            stop_sequences: Sequences that stop generation
            metadata: Additional Anthropic-specific metadata

        Returns:
            ModelResponse with content, tool calls, usage, and cost

        Raises:
            ModelError: If the API call fails
        """
        try:
            # Convert messages from internal format to Anthropic format
            anthropic_messages = self._convert_messages_to_anthropic(messages)

            # Build request parameters
            kwargs: dict[str, Any] = {
                "model": self._model_id,
                "messages": anthropic_messages,
                "temperature": temperature if temperature is not None else self._temperature,
                "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
            }

            if system is not None:
                kwargs["system"] = system

            # Convert tools to Anthropic format (handles ToolManifest objects)
            if tools is not None:
                anthropic_tools = self._convert_tools_to_anthropic(tools)
                if anthropic_tools is not None:
                    kwargs["tools"] = anthropic_tools

            if stop_sequences is not None:
                kwargs["stop_sequences"] = stop_sequences

            if metadata is not None:
                kwargs["metadata"] = metadata

            # Call Anthropic API
            response = await self._client.messages.create(**kwargs)

            # Extract content and tool calls
            content_text = ""
            tool_calls: list[ToolCall] = []

            for block in response.content:
                if block.type == "text":
                    content_text += block.text
                elif block.type == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=block.input,
                        )
                    )

            # Extract exact token counts from response (AD-029)
            usage = NormalizedUsage.from_anthropic(getattr(response, "usage", None))

            # Calculate cost
            cost = self.estimate_cost(usage.input_tokens, usage.output_tokens)

            return ModelResponse(
                content=content_text,
                tool_calls=tuple(tool_calls) if tool_calls else None,
                usage=usage,
                cost_usd=cost,
            )

        except Exception as e:
            reason, recoverable, status_code, provider_code = self._map_anthropic_error(e)
            raise ModelError(
                message=f"Anthropic error for {self._model_id}: {e}",
                model_id=self._model_id,
                reason=reason,
                recoverable=recoverable,
                status_code=status_code,
                provider_code=provider_code,
                details={"error_type": reason},
                cause=e,
            ) from e

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
        """Stream a completion from the model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            system: System prompt/instructions
            tools: Available tools for the model to use
            temperature: Sampling temperature (overrides default)
            max_tokens: Maximum tokens to generate (overrides default)
            stop_sequences: Sequences that stop generation
            metadata: Additional Anthropic-specific metadata

        Yields:
            ModelChunk instances with incremental content/tool calls

        Raises:
            ModelError: If the API call fails
        """
        try:
            # Convert messages from internal format to Anthropic format
            anthropic_messages = self._convert_messages_to_anthropic(messages)

            # Build request parameters
            kwargs: dict[str, Any] = {
                "model": self._model_id,
                "messages": anthropic_messages,
                "temperature": temperature if temperature is not None else self._temperature,
                "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
            }

            if system is not None:
                kwargs["system"] = system

            # Convert tools to Anthropic format (handles ToolManifest objects)
            if tools is not None:
                anthropic_tools = self._convert_tools_to_anthropic(tools)
                if anthropic_tools is not None:
                    kwargs["tools"] = anthropic_tools

            if stop_sequences is not None:
                kwargs["stop_sequences"] = stop_sequences

            if metadata is not None:
                kwargs["metadata"] = metadata

            # Stream from Anthropic API
            async with self._client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, "text"):
                            # Text content delta
                            yield ModelChunk(
                                content=event.delta.text,
                                tool_call_delta=None,
                                is_final=False,
                            )
                        elif hasattr(event.delta, "partial_json"):
                            # Tool use delta
                            yield ModelChunk(
                                content="",
                                tool_call_delta={"partial_json": event.delta.partial_json},
                                is_final=False,
                            )

                    elif event.type == "message_stop":
                        # Final chunk
                        yield ModelChunk(
                            content="",
                            tool_call_delta=None,
                            is_final=True,
                        )

        except Exception as e:
            reason, recoverable, status_code, provider_code = self._map_anthropic_error(e)
            raise ModelError(
                message=f"Anthropic error for {self._model_id}: {e}",
                model_id=self._model_id,
                reason=reason,
                recoverable=recoverable,
                status_code=status_code,
                provider_code=provider_code,
                details={"error_type": reason},
                cause=e,
            ) from e

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses a simple character-based estimation as a fallback.
        For production use, consider using the Anthropic tokenizer library.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count (roughly 1 token per 4 characters)
        """
        if not text:
            return 0

        # Simple estimation: ~4 characters per token
        # This is a rough approximation - actual tokenization varies
        return max(1, len(text) // 4)

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD for token usage.

        Uses the MODEL_PRICING table for known models. Returns 0.0 for unknown models.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        pricing = MODEL_PRICING.get(self._model_id, {})

        if not pricing:
            # Unknown model - return 0 cost
            return 0.0

        input_cost = (input_tokens / 1_000_000) * pricing.get("input_per_mtok", 0)
        output_cost = (output_tokens / 1_000_000) * pricing.get("output_per_mtok", 0)

        return input_cost + output_cost
