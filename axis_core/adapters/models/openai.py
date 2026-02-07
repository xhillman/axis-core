"""OpenAI model adapter implementation.

This module provides the OpenAIModel adapter for GPT models via the OpenAI API.
Requires the 'openai' package: pip install axis-core[openai]
"""

import json
import os
from collections.abc import AsyncIterator
from typing import Any, cast

# Conditional import per AD-040
try:
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletionChunk
except ImportError as e:
    raise ImportError(
        "OpenAIModel requires the openai package. "
        "Install with: pip install axis-core[openai]"
    ) from e

from axis_core.errors import ModelError
from axis_core.protocols.model import ModelChunk, ModelResponse, ToolCall, UsageStats
from axis_core.tool import ToolManifest

# Pricing table for cost estimation (per million tokens)
# Source: https://openai.com/api/pricing/ (as of 2026-01)
MODEL_PRICING: dict[str, dict[str, float]] = {
    # GPT-5 series
    "gpt-5.2": {"input_per_mtok": 1.75, "output_per_mtok": 14.00},
    "gpt-5.1": {"input_per_mtok": 1.25, "output_per_mtok": 10.00},
    "gpt-5": {"input_per_mtok": 1.25, "output_per_mtok": 10.00},
    "gpt-5-mini": {"input_per_mtok": 0.25, "output_per_mtok": 2.00},
    "gpt-5-nano": {"input_per_mtok": 0.05, "output_per_mtok": 0.40},
    "gpt-5.2-chat-latest": {"input_per_mtok": 1.75, "output_per_mtok": 14.00},
    "gpt-5.1-chat-latest": {"input_per_mtok": 1.25, "output_per_mtok": 10.00},
    "gpt-5-chat-latest": {"input_per_mtok": 1.25, "output_per_mtok": 10.00},
    "gpt-5.2-pro": {"input_per_mtok": 21.00, "output_per_mtok": 168.00},
    "gpt-5-pro": {"input_per_mtok": 15.00, "output_per_mtok": 120.00},
    # GPT-4.1 series
    "gpt-4.1": {"input_per_mtok": 2.00, "output_per_mtok": 8.00},
    "gpt-4.1-mini": {"input_per_mtok": 0.40, "output_per_mtok": 1.60},
    "gpt-4.1-nano": {"input_per_mtok": 0.10, "output_per_mtok": 0.40},
    # GPT-4o series
    "gpt-4o": {"input_per_mtok": 2.50, "output_per_mtok": 10.00},
    "gpt-4o-2024-05-13": {"input_per_mtok": 5.00, "output_per_mtok": 15.00},
    "gpt-4o-mini": {"input_per_mtok": 0.15, "output_per_mtok": 0.60},
    # O-series reasoning models
    "o1": {"input_per_mtok": 15.00, "output_per_mtok": 60.00},
    "o1-pro": {"input_per_mtok": 150.00, "output_per_mtok": 600.00},
    "o1-mini": {"input_per_mtok": 1.10, "output_per_mtok": 4.40},
    "o3": {"input_per_mtok": 2.00, "output_per_mtok": 8.00},
    "o3-pro": {"input_per_mtok": 20.00, "output_per_mtok": 80.00},
    "o3-mini": {"input_per_mtok": 1.10, "output_per_mtok": 4.40},
    "o4-mini": {"input_per_mtok": 1.10, "output_per_mtok": 4.40},
}



class OpenAIModel:
    """OpenAI GPT model adapter.

    Provides access to GPT models through the OpenAI Chat Completions API.
    Supports streaming, function calling, and cost tracking.

    Args:
        model_id: Model identifier (e.g., 'gpt-4', 'gpt-4o', 'gpt-3.5-turbo')
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        temperature: Sampling temperature (0.0-2.0, default 1.0)
        max_tokens: Maximum tokens to generate (default 4096)

    Example:
        >>> model = OpenAIModel(model_id="gpt-4")
        >>> response = await model.complete(
        ...     messages=[{"role": "user", "content": "Hello"}],
        ...     system="You are a helpful assistant."
        ... )
        >>> print(response.content)
        Hello! How can I help you today?
    """

    # Models that use max_completion_tokens instead of max_tokens
    _COMPLETION_TOKENS_MODELS = {
        "gpt-5.2",
        "gpt-5.1",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-5.2-chat-latest",
        "gpt-5.1-chat-latest",
        "gpt-5-chat-latest",
        "gpt-5.2-codex",
        "gpt-5.1-codex-max",
        "gpt-5.1-codex",
        "gpt-5-codex",
        "gpt-5.1-codex-mini",
        "codex-mini-latest",
        "gpt-5.2-pro",
        "gpt-5-pro",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "o1",
        "o1-pro",
        "o1-mini",
        "o3",
        "o3-pro",
        "o3-mini",
        "o3-deep-research",
        "o4-mini",
        "o4-mini-deep-research",
        "gpt-5-search-api",
    }

    def __init__(
        self,
        model_id: str,
        api_key: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
    ) -> None:
        """Initialize the OpenAI model adapter.

        Args:
            model_id: Model identifier
            api_key: API key (defaults to OPENAI_API_KEY env var)
            temperature: Default temperature for completions
            max_tokens: Default max tokens for completions

        Raises:
            ValueError: If api_key is not provided and OPENAI_API_KEY is not set
        """
        self._model_id = model_id
        self._temperature = temperature
        self._max_tokens = max_tokens

        # Get API key from parameter or environment
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Provide it via api_key parameter or set OPENAI_API_KEY environment variable."
            )

        # Initialize OpenAI client
        self._client = AsyncOpenAI(api_key=self._api_key)

    def _uses_completion_tokens(self) -> bool:
        """Check if this model uses max_completion_tokens instead of max_tokens."""
        return self._model_id in self._COMPLETION_TOKENS_MODELS

    @property
    def model_id(self) -> str:
        """Return the model identifier."""
        return self._model_id

    @staticmethod
    def _convert_tool_manifest_to_openai(manifest: ToolManifest) -> dict[str, Any]:
        """Convert a ToolManifest to OpenAI's function calling format.

        Args:
            manifest: Tool manifest from the protocol layer

        Returns:
            Dict in OpenAI's function calling format

        Example:
            >>> manifest = ToolManifest(
            ...     name="get_weather",
            ...     description="Get weather",
            ...     input_schema={"type": "object", "properties": {...}},
            ...     ...
            ... )
            >>> schema = OpenAIModel._convert_tool_manifest_to_openai(manifest)
            >>> schema["type"]
            "function"
        """
        return {
            "type": "function",
            "function": {
                "name": manifest.name,
                "description": manifest.description,
                "parameters": manifest.input_schema,
            },
        }

    @staticmethod
    def _convert_messages_to_openai(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert messages from internal format to OpenAI's API format.

        Ensures that tool_calls in assistant messages have the required 'type' field.

        Args:
            messages: List of messages in internal format

        Returns:
            List of messages in OpenAI's API format
        """
        converted: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")

            # If assistant message has tool_calls, ensure they have type field
            if role == "assistant" and "tool_calls" in msg:
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    # Ensure each tool_call has the required 'type' field
                    formatted_calls = []
                    for tc in tool_calls:
                        # If already properly formatted, pass through
                        if isinstance(tc, dict) and "type" in tc:
                            formatted_calls.append(tc)
                        # Otherwise, add the type field
                        elif isinstance(tc, dict):
                            formatted_calls.append({
                                "id": tc.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": tc.get("name", ""),
                                    "arguments": (
                                        json.dumps(tc.get("arguments", {}))
                                        if isinstance(tc.get("arguments"), dict)
                                        else tc.get("arguments", "{}")
                                    ),
                                },
                            })

                    # Create converted message with formatted tool_calls
                    converted_msg = {
                        "role": "assistant",
                        "content": msg.get("content", ""),
                        "tool_calls": formatted_calls,
                    }
                    converted.append(converted_msg)
                else:
                    # No tool_calls, pass through
                    converted.append(msg)
            else:
                # Pass through non-assistant messages or assistant without tool_calls
                converted.append(msg)

        return converted

    @staticmethod
    def _convert_tools_to_openai(tools: Any) -> list[dict[str, Any]] | None:
        """Convert tools parameter to OpenAI format.

        Handles both ToolManifest objects (protocol layer) and raw dicts
        (for backward compatibility or direct API usage).

        Args:
            tools: List of ToolManifest objects, list of dicts, or None

        Returns:
            List of tool dicts in OpenAI format, or None if no tools
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
                tool_dicts: list[dict[str, Any]] = tools
                return tool_dicts

            # If ToolManifest objects, convert them
            if isinstance(first_tool, ToolManifest):
                return [
                    OpenAIModel._convert_tool_manifest_to_openai(manifest) for manifest in tools
                ]

        # Shouldn't reach here with proper input types
        return None

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
            metadata: Additional metadata (not used by OpenAI API)

        Returns:
            ModelResponse with content, tool calls, usage, and cost

        Raises:
            ModelError: If the API call fails
        """
        try:
            # Convert messages from internal format to OpenAI format
            openai_messages = self._convert_messages_to_openai(messages)

            # Build messages list - prepend system message if provided
            api_messages = []
            if system is not None:
                api_messages.append({"role": "system", "content": system})
            api_messages.extend(openai_messages)

            # Build request parameters
            kwargs: dict[str, Any] = {
                "model": self._model_id,
                "messages": api_messages,
                "temperature": temperature if temperature is not None else self._temperature,
            }

            # Use max_completion_tokens for newer models, max_tokens for older ones
            if max_tokens is not None or self._max_tokens is not None:
                tokens_value = max_tokens if max_tokens is not None else self._max_tokens
                if self._uses_completion_tokens():
                    kwargs["max_completion_tokens"] = tokens_value
                else:
                    kwargs["max_tokens"] = tokens_value

            # Convert tools to OpenAI format (handles ToolManifest objects)
            if tools is not None:
                openai_tools = self._convert_tools_to_openai(tools)
                if openai_tools is not None:
                    kwargs["tools"] = openai_tools

            if stop_sequences is not None:
                kwargs["stop"] = stop_sequences

            # Call OpenAI API
            response = await self._client.chat.completions.create(**kwargs)

            # Extract content and tool calls
            message = response.choices[0].message
            content_text = message.content or ""
            tool_calls_list: list[ToolCall] = []

            if message.tool_calls:
                for tc in message.tool_calls:
                    # Parse arguments from JSON string
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        # Fallback to empty dict if arguments are invalid JSON
                        arguments = {}

                    tool_calls_list.append(
                        ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=arguments,
                        )
                    )

            # Extract exact token counts from response
            usage = UsageStats.from_openai(
                {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            )

            # Calculate cost
            cost = self.estimate_cost(usage.input_tokens, usage.output_tokens)

            return ModelResponse(
                content=content_text,
                tool_calls=tuple(tool_calls_list) if tool_calls_list else None,
                usage=usage,
                cost_usd=cost,
            )

        except Exception as e:
            # Import OpenAI exceptions locally to handle them
            import openai

            if isinstance(e, openai.RateLimitError):
                # Recoverable error - can be retried
                raise ModelError(
                    message=f"Rate limit exceeded for {self._model_id}: {e}",
                    model_id=self._model_id,
                    recoverable=True,
                    details={"error_type": "rate_limit"},
                    cause=e,
                ) from e

            elif isinstance(e, openai.AuthenticationError):
                # Non-recoverable error - bad API key
                raise ModelError(
                    message=f"Authentication failed for {self._model_id}: {e}",
                    model_id=self._model_id,
                    recoverable=False,
                    details={"error_type": "authentication"},
                    cause=e,
                ) from e

            elif isinstance(e, openai.BadRequestError):
                # Non-recoverable error - invalid request
                raise ModelError(
                    message=f"Bad request to {self._model_id}: {e}",
                    model_id=self._model_id,
                    recoverable=False,
                    details={"error_type": "bad_request"},
                    cause=e,
                ) from e

            elif isinstance(e, openai.APITimeoutError):
                # Recoverable error - timeout
                raise ModelError(
                    message=f"API timeout for {self._model_id}: {e}",
                    model_id=self._model_id,
                    recoverable=True,
                    details={"error_type": "timeout"},
                    cause=e,
                ) from e

            elif isinstance(e, openai.APIError):
                # Generic API error - assume non-recoverable unless proven otherwise
                raise ModelError(
                    message=f"API error for {self._model_id}: {e}",
                    model_id=self._model_id,
                    recoverable=False,
                    details={"error_type": "api_error"},
                    cause=e,
                ) from e

            else:
                # Unknown error - wrap it
                raise ModelError(
                    message=f"Unexpected error for {self._model_id}: {e}",
                    model_id=self._model_id,
                    recoverable=False,
                    details={"error_type": "unknown"},
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
            metadata: Additional metadata (not used by OpenAI API)

        Yields:
            ModelChunk instances with incremental content/tool calls

        Raises:
            ModelError: If the API call fails
        """
        try:
            # Convert messages from internal format to OpenAI format
            openai_messages = self._convert_messages_to_openai(messages)

            # Build messages list - prepend system message if provided
            api_messages = []
            if system is not None:
                api_messages.append({"role": "system", "content": system})
            api_messages.extend(openai_messages)

            # Build request parameters
            kwargs: dict[str, Any] = {
                "model": self._model_id,
                "messages": api_messages,
                "temperature": temperature if temperature is not None else self._temperature,
                "stream": True,
            }

            # Use max_completion_tokens for newer models, max_tokens for older ones
            if max_tokens is not None or self._max_tokens is not None:
                tokens_value = max_tokens if max_tokens is not None else self._max_tokens
                if self._uses_completion_tokens():
                    kwargs["max_completion_tokens"] = tokens_value
                else:
                    kwargs["max_tokens"] = tokens_value

            # Convert tools to OpenAI format
            if tools is not None:
                openai_tools = self._convert_tools_to_openai(tools)
                if openai_tools is not None:
                    kwargs["tools"] = openai_tools

            if stop_sequences is not None:
                kwargs["stop"] = stop_sequences

            # Stream from OpenAI API
            # cast() is needed because mypy can't resolve the create() overload
            # when stream=True is passed via **kwargs (the SDK uses Literal[True]
            # in its overload signatures, which requires a static keyword).
            stream = cast(
                AsyncIterator[ChatCompletionChunk],
                self._client.chat.completions.create(**kwargs),
            )
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta

                    # Check if this is content delta
                    if delta.content is not None:
                        yield ModelChunk(
                            content=delta.content,
                            tool_call_delta=None,
                            is_final=False,
                        )

                    # Check if this is tool call delta
                    elif delta.tool_calls is not None:
                        # Tool calls are being streamed
                        for tc_delta in delta.tool_calls:
                            tool_delta: dict[str, Any] = {
                                "index": tc_delta.index,
                            }
                            if hasattr(tc_delta, "id") and tc_delta.id:
                                tool_delta["id"] = tc_delta.id
                            if hasattr(tc_delta, "function") and tc_delta.function:
                                tool_delta["function"] = {
                                    "name": tc_delta.function.name,
                                    "arguments": tc_delta.function.arguments,
                                }

                            yield ModelChunk(
                                content="",
                                tool_call_delta=tool_delta,
                                is_final=False,
                            )

                # Check for stream end
                if chunk.choices and chunk.choices[0].finish_reason is not None:
                    yield ModelChunk(
                        content="",
                        tool_call_delta=None,
                        is_final=True,
                    )

        except Exception as e:
            import openai

            if isinstance(e, openai.RateLimitError):
                raise ModelError(
                    message=f"Rate limit exceeded for {self._model_id}: {e}",
                    model_id=self._model_id,
                    recoverable=True,
                    details={"error_type": "rate_limit"},
                    cause=e,
                ) from e

            elif isinstance(e, openai.AuthenticationError):
                raise ModelError(
                    message=f"Authentication failed for {self._model_id}: {e}",
                    model_id=self._model_id,
                    recoverable=False,
                    details={"error_type": "authentication"},
                    cause=e,
                ) from e

            elif isinstance(e, openai.APIError):
                raise ModelError(
                    message=f"API error for {self._model_id}: {e}",
                    model_id=self._model_id,
                    recoverable=False,
                    details={"error_type": "api_error"},
                    cause=e,
                ) from e

            else:
                raise ModelError(
                    message=f"Unexpected error for {self._model_id}: {e}",
                    model_id=self._model_id,
                    recoverable=False,
                    details={"error_type": "unknown"},
                    cause=e,
                ) from e

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses a simple character-based estimation as a fallback.
        For production use, consider using the tiktoken library.

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
