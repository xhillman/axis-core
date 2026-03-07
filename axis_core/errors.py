"""Error handling system for axis-core.

Provides ErrorClass enum, AxisError base class, specific error types,
and ErrorRecord for tracking errors throughout agent execution.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import ClassVar


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


@dataclass(frozen=True)
class NormalizedProviderError:
    """Shared normalized metadata for provider-backed model failures."""

    reason: str
    recoverable: bool
    status_code: int | None
    provider_code: str | None


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
    reason: str = "unknown"
    status_code: int | None = None
    provider_code: str | None = None
    error_class: ErrorClass = field(default=ErrorClass.MODEL, init=False)

    _RECOVERABLE_REASONS: ClassVar[frozenset[str]] = frozenset(
        {
            "rate_limit",
            "timeout",
            "connection_error",
            "service_unavailable",
            "transient_provider_error",
        }
    )

    @classmethod
    def is_reason_recoverable(cls, reason: str | None) -> bool:
        """Return True when a normalized reason is retry/fallback eligible."""
        return (reason or "unknown") in cls._RECOVERABLE_REASONS

    @staticmethod
    def _extract_status_code(e: Exception) -> int | None:
        """Best-effort extraction of HTTP status code from provider errors."""
        raw_status = getattr(e, "status_code", None)
        if isinstance(raw_status, int):
            return raw_status

        response = getattr(e, "response", None)
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int):
            return response_status

        return None

    @staticmethod
    def _extract_provider_code(e: Exception) -> str | None:
        """Best-effort extraction of provider-specific error code."""
        direct_code = getattr(e, "code", None)
        if isinstance(direct_code, str) and direct_code:
            return direct_code

        body = getattr(e, "body", None)
        if isinstance(body, dict):
            body_code = body.get("code")
            if isinstance(body_code, str) and body_code:
                return body_code
            error_obj = body.get("error")
            if isinstance(error_obj, dict):
                nested_code = error_obj.get("code")
                if isinstance(nested_code, str) and nested_code:
                    return nested_code
                nested_type = error_obj.get("type")
                if isinstance(nested_type, str) and nested_type:
                    return nested_type

        return None

    @staticmethod
    def reason_from_status_code(status_code: int | None) -> str | None:
        """Map common HTTP status codes to normalized failure reasons."""
        if status_code is None:
            return None
        if status_code == 429:
            return "rate_limit"
        if status_code in {408, 504}:
            return "timeout"
        if status_code in {500, 502, 503}:
            return "service_unavailable"
        if status_code in {401, 403}:
            return "authentication"
        if status_code in {400, 404, 409, 410, 413, 415, 422}:
            return "invalid_request"
        return None

    @classmethod
    def normalize_provider_error(
        cls,
        e: Exception,
        *,
        explicit_reason: str | None = None,
        provider_error_reason: str | None = None,
    ) -> NormalizedProviderError:
        """Normalize shared provider error metadata behind one boundary."""
        status_code = cls._extract_status_code(e)
        provider_code = cls._extract_provider_code(e)
        reason = (
            explicit_reason
            or cls.reason_from_status_code(status_code)
            or provider_error_reason
            or "unknown"
        )
        return NormalizedProviderError(
            reason=reason,
            recoverable=cls.is_reason_recoverable(reason),
            status_code=status_code,
            provider_code=provider_code,
        )

    @classmethod
    def build_provider_error(
        cls,
        e: Exception,
        *,
        model_id: str,
        message: str,
        explicit_reason: str | None = None,
        provider_error_reason: str | None = None,
        details: dict[str, object] | None = None,
    ) -> "ModelError":
        """Build a ModelError from shared normalized provider metadata."""
        normalized = cls.normalize_provider_error(
            e,
            explicit_reason=explicit_reason,
            provider_error_reason=provider_error_reason,
        )
        error_details: dict[str, object] = {"error_type": normalized.reason}
        if details:
            error_details.update(details)
        return cls(
            message=message,
            model_id=model_id,
            reason=normalized.reason,
            recoverable=normalized.recoverable,
            status_code=normalized.status_code,
            provider_code=normalized.provider_code,
            details=error_details,
            cause=e,
        )

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

        recoverable_reason_by_name = {
            "RateLimitError": "rate_limit",
            "TimeoutError": "timeout",
            "ConnectError": "connection_error",
            "ConnectionError": "connection_error",
            "ReadTimeout": "timeout",
            "ReadTimeoutError": "timeout",
            "APIConnectionError": "connection_error",
            "APITimeoutError": "timeout",
        }

        non_recoverable_reason_by_name = {
            "ValidationError": "invalid_request",
            "BadRequestError": "invalid_request",
            "AuthenticationError": "authentication",
            "AuthError": "authentication",
            "TypeError": "invalid_request",
            "ValueError": "invalid_request",
            "KeyError": "invalid_request",
        }

        explicit_reason = recoverable_reason_by_name.get(exc_class_name)
        if explicit_reason is None:
            explicit_reason = non_recoverable_reason_by_name.get(exc_class_name)

        return cls.build_provider_error(
            e,
            model_id=model_id,
            message=f"Model error: {str(e)}",
            explicit_reason=explicit_reason,
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
