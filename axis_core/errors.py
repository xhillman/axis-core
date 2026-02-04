"""Error handling system for axis-core.

Provides ErrorClass enum, AxisError base class, specific error types,
and ErrorRecord for tracking errors throughout agent execution.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ErrorClass(Enum):
    """Classification of error types in the system."""

    INPUT = "input"
    CONFIG = "config"
    PLAN = "plan"
    TOOL = "tool"
    MODEL = "model"
    BUDGET = "budget"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    RUNTIME = "runtime"


@dataclass
class AxisError(Exception):
    """Base exception for all axis-core errors.

    All errors in the system inherit from this class, providing consistent
    error handling with rich context and recovery information.
    """

    message: str
    error_class: ErrorClass
    phase: str | None = None
    cycle: int | None = None
    step_id: str | None = None
    recoverable: bool = False
    retry_after: float | None = None
    details: dict[str, object] = field(default_factory=dict)
    cause: Exception | None = None

    def __post_init__(self) -> None:
        """Initialize the Exception base class with the message."""
        super().__init__(self.message)


@dataclass
class InputError(AxisError):
    """Error in user input or task specification."""

    error_class: ErrorClass = field(default=ErrorClass.INPUT, init=False)


@dataclass
class ConfigError(AxisError):
    """Error in agent configuration or setup."""

    error_class: ErrorClass = field(default=ErrorClass.CONFIG, init=False)


@dataclass
class PlanError(AxisError):
    """Error during planning phase."""

    error_class: ErrorClass = field(default=ErrorClass.PLAN, init=False)


@dataclass
class TimeoutError(AxisError):
    """Error when operation exceeds time limit."""

    error_class: ErrorClass = field(default=ErrorClass.TIMEOUT, init=False)


@dataclass
class CancelledError(AxisError):
    """Error when operation is cancelled."""

    error_class: ErrorClass = field(default=ErrorClass.CANCELLED, init=False)


@dataclass
class ConcurrencyError(AxisError):
    """Error when optimistic concurrency checks fail."""

    expected_version: int | None = None
    actual_version: int | None = None
    error_class: ErrorClass = field(default=ErrorClass.RUNTIME, init=False)


@dataclass
class ToolError(AxisError):
    """Error during tool execution."""

    tool_name: str | None = None
    error_class: ErrorClass = field(default=ErrorClass.TOOL, init=False)


@dataclass
class ModelError(AxisError):
    """Error from model adapter or LLM provider.

    Includes logic to classify exceptions from various LLM SDKs as
    recoverable (worth retrying) or not recoverable (permanent failure).
    """

    model_id: str | None = None
    error_class: ErrorClass = field(default=ErrorClass.MODEL, init=False)

    @classmethod
    def from_exception(cls, e: Exception, model_id: str) -> "ModelError":
        """Create ModelError from an exception, classifying recoverability.

        Recoverable errors (worth retrying):
        - RateLimitError, TimeoutError, ConnectionError, ConnectError
        - ReadTimeout, ReadTimeoutError, APIConnectionError

        Not recoverable (permanent failures):
        - ValidationError, AuthenticationError, TypeError, ValueError, KeyError
        - Unknown exceptions default to not recoverable

        Args:
            e: Original exception from model provider
            model_id: Identifier of the model that failed

        Returns:
            ModelError with appropriate recoverability flag
        """
        exc_class_name = type(e).__name__

        # Check if exception is recoverable based on class name
        recoverable_errors = {
            "RateLimitError",
            "TimeoutError",
            "ConnectionError",
            "ConnectError",
            "ReadTimeout",
            "ReadTimeoutError",
            "APIConnectionError",
        }

        non_recoverable_errors = {
            "ValidationError",
            "AuthenticationError",
            "AuthError",
            "TypeError",
            "ValueError",
            "KeyError",
        }

        is_recoverable = False
        if exc_class_name in recoverable_errors:
            is_recoverable = True
        elif exc_class_name in non_recoverable_errors:
            is_recoverable = False
        # Unknown errors default to not recoverable

        return cls(
            message=f"Model error: {str(e)}",
            model_id=model_id,
            recoverable=is_recoverable,
            cause=e,
        )


@dataclass
class BudgetError(AxisError):
    """Error when budget limit is exceeded.

    Automatically generates actionable suggestions for adjusting budget
    parameters based on actual usage.
    """

    resource: str | None = None
    used: float | None = None
    limit: float | None = None
    error_class: ErrorClass = field(default=ErrorClass.BUDGET, init=False)

    def __post_init__(self) -> None:
        """Generate rich error message with actionable suggestion."""
        # Generate suggestion if we have resource info
        if self.resource is not None and self.used is not None and self.limit is not None:
            # Map resource names to parameter names
            param_map = {
                "cost_usd": "budget.max_cost_usd",
                "input_tokens": "budget.max_input_tokens",
                "output_tokens": "budget.max_output_tokens",
                "cycles": "budget.max_cycles",
            }

            param_name = param_map.get(self.resource, f"budget.{self.resource}")

            # Suggest ~1.5x the used value
            suggested_value = self.used * 1.5

            # Format based on resource type
            if self.resource == "cost_usd":
                suggestion = (
                    f"\n\nSuggestion: Increase {param_name} to {suggested_value:.2f} "
                    f"(current: {self.limit}, used: {self.used:.2f})"
                )
            elif isinstance(self.used, int):
                suggestion = (
                    f"\n\nSuggestion: Increase {param_name} to {int(suggested_value)} "
                    f"(current: {int(self.limit)}, used: {int(self.used)})"
                )
            else:
                suggestion = (
                    f"\n\nSuggestion: Increase {param_name} to {suggested_value} "
                    f"(current: {self.limit}, used: {self.used})"
                )

            self.message += suggestion

        # Call parent __post_init__ to initialize Exception
        super().__post_init__()


@dataclass(frozen=True)
class ErrorRecord:
    """Immutable record of an error that occurred during execution.

    Used to track error history in the agent's execution context.
    Per AD-014, error records are immutable (append-only history).
    """

    error: AxisError
    timestamp: datetime
    phase: str
    cycle: int
    recovered: bool
