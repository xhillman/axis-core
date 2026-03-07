"""Shared OpenAI error normalization utilities."""

from __future__ import annotations

from axis_core.errors import ModelError

try:
    import openai
except ImportError as e:
    raise ImportError(
        "OpenAI adapters require the openai package. "
        "Install with: pip install axis-core[openai]"
    ) from e


def _resolve_openai_reason(error: Exception) -> tuple[str | None, str | None]:
    """Keep OpenAI exception-type matching close to the OpenAI adapters."""
    permission_error = getattr(openai, "PermissionDeniedError", ())
    unprocessable_error = getattr(openai, "UnprocessableEntityError", ())
    bad_request_types = tuple(
        t for t in (openai.BadRequestError, unprocessable_error) if t
    )
    auth_types = tuple(
        t for t in (openai.AuthenticationError, permission_error) if t
    )

    if isinstance(error, openai.RateLimitError):
        return "rate_limit", None
    if isinstance(error, openai.APITimeoutError):
        return "timeout", None
    if isinstance(error, openai.APIConnectionError):
        return "connection_error", None
    if auth_types and isinstance(error, auth_types):
        return "authentication", None
    if bad_request_types and isinstance(error, bad_request_types):
        return "invalid_request", None
    if isinstance(error, (openai.APIStatusError, openai.APIError)):
        return None, "provider_error"
    return None, None


def classify_openai_error(error: Exception) -> tuple[str, bool, int | None, str | None]:
    """Normalize OpenAI SDK exceptions into reason-coded fallback metadata."""
    explicit_reason, provider_error_reason = _resolve_openai_reason(error)
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


def build_openai_model_error(error: Exception, model_id: str) -> ModelError:
    """Build a normalized ModelError from an OpenAI SDK exception."""
    explicit_reason, provider_error_reason = _resolve_openai_reason(error)
    return ModelError.build_provider_error(
        error,
        model_id=model_id,
        message=f"OpenAI error for {model_id}: {error}",
        explicit_reason=explicit_reason,
        provider_error_reason=provider_error_reason,
    )
