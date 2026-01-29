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

from axis_core.errors import ModelError
from axis_core.protocols.model import ModelChunk, ModelResponse, ToolCall, UsageStats

# Pricing table for cost estimation (per million tokens)
# Source: https://platform.claude.com/docs/en/about-claude/pricing (as of 2026-01)
MODEL_PRICING: dict[str, dict[str, float]] = {
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
            tools: Available tools for the model to use
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
            # Build request parameters
            kwargs: dict[str, Any] = {
                "model": self._model_id,
                "messages": messages,
                "temperature": temperature if temperature is not None else self._temperature,
                "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
            }

            if system is not None:
                kwargs["system"] = system

            if tools is not None:
                kwargs["tools"] = tools

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
            usage = UsageStats.from_anthropic(
                {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
            )

            # Calculate cost
            cost = self.estimate_cost(usage.input_tokens, usage.output_tokens)

            return ModelResponse(
                content=content_text,
                tool_calls=tuple(tool_calls) if tool_calls else None,
                usage=usage,
                cost_usd=cost,
            )

        except anthropic.RateLimitError as e:
            # Recoverable error - can be retried (AD-013)
            raise ModelError(
                message=f"Rate limit exceeded for {self._model_id}: {e}",
                model_id=self._model_id,
                recoverable=True,
                details={"error_type": "rate_limit"},
                cause=e,
            ) from e

        except anthropic.AuthenticationError as e:
            # Non-recoverable error - bad API key
            raise ModelError(
                message=f"Authentication failed for {self._model_id}: {e}",
                model_id=self._model_id,
                recoverable=False,
                details={"error_type": "authentication"},
                cause=e,
            ) from e

        except anthropic.BadRequestError as e:
            # Non-recoverable error - invalid request
            raise ModelError(
                message=f"Bad request to {self._model_id}: {e}",
                model_id=self._model_id,
                recoverable=False,
                details={"error_type": "bad_request"},
                cause=e,
            ) from e

        except anthropic.APITimeoutError as e:
            # Recoverable error - timeout
            raise ModelError(
                message=f"API timeout for {self._model_id}: {e}",
                model_id=self._model_id,
                recoverable=True,
                details={"error_type": "timeout"},
                cause=e,
            ) from e

        except anthropic.APIError as e:
            # Generic API error - assume non-recoverable unless proven otherwise
            raise ModelError(
                message=f"API error for {self._model_id}: {e}",
                model_id=self._model_id,
                recoverable=False,
                details={"error_type": "api_error"},
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
            # Build request parameters
            kwargs: dict[str, Any] = {
                "model": self._model_id,
                "messages": messages,
                "temperature": temperature if temperature is not None else self._temperature,
                "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
            }

            if system is not None:
                kwargs["system"] = system

            if tools is not None:
                kwargs["tools"] = tools

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

        except anthropic.RateLimitError as e:
            raise ModelError(
                message=f"Rate limit exceeded for {self._model_id}: {e}",
                model_id=self._model_id,
                recoverable=True,
                details={"error_type": "rate_limit"},
                cause=e,
            ) from e

        except anthropic.AuthenticationError as e:
            raise ModelError(
                message=f"Authentication failed for {self._model_id}: {e}",
                model_id=self._model_id,
                recoverable=False,
                details={"error_type": "authentication"},
                cause=e,
            ) from e

        except anthropic.APIError as e:
            raise ModelError(
                message=f"API error for {self._model_id}: {e}",
                model_id=self._model_id,
                recoverable=False,
                details={"error_type": "api_error"},
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
