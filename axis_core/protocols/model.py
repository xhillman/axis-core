"""Model adapter protocol and associated dataclasses.

This module defines the ModelAdapter protocol interface for LLM providers, along with
dataclasses for representing model responses, tool calls, and usage statistics.
"""

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class NormalizedUsage:
    """Token usage statistics from a model call.

    Attributes:
        input_tokens: Number of tokens in the input/prompt
        output_tokens: Number of tokens in the output/completion
        total_tokens: Total tokens used (input + output)
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int

    @staticmethod
    def _read_field(payload: Any, key: str) -> Any:
        """Read a usage field from either mapping or object payloads."""
        if payload is None:
            return None
        if isinstance(payload, Mapping):
            return payload.get(key)
        return getattr(payload, key, None)

    @staticmethod
    def _coerce_non_negative_int(value: Any) -> int:
        """Best-effort conversion to a non-negative int token count."""
        coerced, is_valid = NormalizedUsage._try_coerce_non_negative_int(value)
        if not is_valid:
            return 0
        return coerced

    @staticmethod
    def _try_coerce_non_negative_int(value: Any) -> tuple[int, bool]:
        """Attempt conversion to non-negative int with validity signal."""
        if value is None:
            return 0, False

        try:
            coerced = int(value)
        except (TypeError, ValueError):
            return 0, False

        return max(0, coerced), True

    @staticmethod
    def from_anthropic(usage: Any) -> "NormalizedUsage":
        """Create NormalizedUsage from Anthropic API usage object.

        Args:
            usage: Usage payload from Anthropic response
                (for example {"input_tokens": 10, "output_tokens": 20})

        Returns:
            NormalizedUsage instance

        Examples:
            >>> usage = {"input_tokens": 100, "output_tokens": 50}
            >>> stats = NormalizedUsage.from_anthropic(usage)
            >>> stats.total_tokens
            150
        """
        input_tokens = NormalizedUsage._coerce_non_negative_int(
            NormalizedUsage._read_field(usage, "input_tokens")
        )
        output_tokens = NormalizedUsage._coerce_non_negative_int(
            NormalizedUsage._read_field(usage, "output_tokens")
        )
        return NormalizedUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

    @staticmethod
    def from_openai(usage: Any) -> "NormalizedUsage":
        """Create NormalizedUsage from OpenAI API usage object.

        Args:
            usage: Usage payload from OpenAI response
                (for example {"prompt_tokens": 10, "completion_tokens": 20,
                "total_tokens": 30})

        Returns:
            NormalizedUsage instance

        Examples:
            >>> usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
            >>> stats = NormalizedUsage.from_openai(usage)
            >>> stats.input_tokens
            100
        """
        input_tokens, input_valid = NormalizedUsage._try_coerce_non_negative_int(
            NormalizedUsage._read_field(usage, "prompt_tokens")
        )
        if not input_valid:
            input_tokens = NormalizedUsage._coerce_non_negative_int(
                NormalizedUsage._read_field(usage, "input_tokens")
            )

        output_tokens, output_valid = NormalizedUsage._try_coerce_non_negative_int(
            NormalizedUsage._read_field(usage, "completion_tokens")
        )
        if not output_valid:
            output_tokens = NormalizedUsage._coerce_non_negative_int(
                NormalizedUsage._read_field(usage, "output_tokens")
            )

        raw_total = NormalizedUsage._read_field(usage, "total_tokens")
        total_tokens = NormalizedUsage._coerce_non_negative_int(raw_total)
        summed_tokens = input_tokens + output_tokens
        if raw_total is None or total_tokens < summed_tokens:
            total_tokens = summed_tokens

        return NormalizedUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )


# Backward-compatible alias kept for existing imports and public API stability.
UsageStats = NormalizedUsage


@dataclass(frozen=True)
class ToolCall:
    """Represents a tool call from the model.

    Attributes:
        id: Unique identifier for this tool call
        name: Name of the tool to invoke
        arguments: Tool arguments as a dict
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelResponse:
    """Complete response from a model completion.

    Attributes:
        content: Text content of the response
        tool_calls: Tuple of tool calls requested by the model (None if no tools)
        usage: Token usage statistics
        cost_usd: Estimated cost in USD for this completion
    """

    content: str
    tool_calls: tuple[ToolCall, ...] | None
    usage: NormalizedUsage
    cost_usd: float


@dataclass(frozen=True)
class ModelChunk:
    """A streaming chunk from a model stream.

    Attributes:
        content: Incremental text content (empty string if no content)
        tool_call_delta: Incremental tool call data (None if no tool call)
        is_final: Whether this is the final chunk in the stream
    """

    content: str = ""
    tool_call_delta: dict[str, Any] | None = None
    is_final: bool = False


@runtime_checkable
class ModelAdapter(Protocol):
    """Protocol for LLM model adapters.

    Model adapters provide a uniform interface to different LLM providers (Anthropic, OpenAI,
    Ollama, etc.). They handle API calls, token counting, and cost estimation.

    Implementations must provide:
    - model_id property returning the model identifier
    - complete() for non-streaming completions
    - stream() for streaming completions
    - estimate_tokens() for token counting
    - estimate_cost() for cost calculation
    """

    @property
    def model_id(self) -> str:
        """Model identifier (e.g., 'claude-sonnet-4-20250514', 'gpt-4')."""
        ...

    async def complete(
        self,
        messages: Any,  # list[Message]
        system: str | None = None,
        tools: Any | None = None,  # list[Tool] | None
        temperature: float = 1.0,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ModelResponse:
        """Complete a prompt with the model (non-streaming).

        Args:
            messages: List of messages in the conversation
            system: System prompt/instructions
            tools: Available tools for the model to use
            temperature: Sampling temperature (0.0 = deterministic, 2.0 = very random)
            max_tokens: Maximum tokens to generate (None = model default)
            stop_sequences: Sequences that stop generation
            metadata: Additional provider-specific metadata

        Returns:
            ModelResponse with content, tool calls, usage, and cost
        """
        ...

    async def stream(
        self,
        messages: Any,  # list[Message]
        system: str | None = None,
        tools: Any | None = None,  # list[Tool] | None
        temperature: float = 1.0,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[ModelChunk]:
        """Stream a completion from the model.

        Args:
            messages: List of messages in the conversation
            system: System prompt/instructions
            tools: Available tools for the model to use
            temperature: Sampling temperature (0.0 = deterministic, 2.0 = very random)
            max_tokens: Maximum tokens to generate (None = model default)
            stop_sequences: Sequences that stop generation
            metadata: Additional provider-specific metadata

        Yields:
            ModelChunk instances with incremental content/tool calls
        """
        ...

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        ...

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD for token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        ...
