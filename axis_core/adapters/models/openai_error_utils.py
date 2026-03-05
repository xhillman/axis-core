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


def _extract_status_code(error: Exception) -> int | None:
    """Extract status code from OpenAI SDK exception objects."""
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    response = getattr(error, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int):
        return response_status

    return None


def _extract_provider_code(error: Exception) -> str | None:
    """Extract provider error code from OpenAI exception payload."""
    direct_code = getattr(error, "code", None)
    if isinstance(direct_code, str) and direct_code:
        return direct_code

    body = getattr(error, "body", None)
    if isinstance(body, dict):
        top_code = body.get("code")
        if isinstance(top_code, str) and top_code:
            return top_code
        nested = body.get("error")
        if isinstance(nested, dict):
            nested_code = nested.get("code")
            if isinstance(nested_code, str) and nested_code:
                return nested_code

    return None


def classify_openai_error(error: Exception) -> tuple[str, bool, int | None, str | None]:
    """Normalize OpenAI SDK exceptions into reason-coded fallback metadata."""
    status_code = _extract_status_code(error)
    provider_code = _extract_provider_code(error)

    permission_error = getattr(openai, "PermissionDeniedError", ())
    unprocessable_error = getattr(openai, "UnprocessableEntityError", ())
    bad_request_types = tuple(
        t for t in (openai.BadRequestError, unprocessable_error) if t
    )
    auth_types = tuple(
        t for t in (openai.AuthenticationError, permission_error) if t
    )

    if isinstance(error, openai.RateLimitError):
        reason = "rate_limit"
    elif isinstance(error, openai.APITimeoutError):
        reason = "timeout"
    elif isinstance(error, openai.APIConnectionError):
        reason = "connection_error"
    elif auth_types and isinstance(error, auth_types):
        reason = "authentication"
    elif bad_request_types and isinstance(error, bad_request_types):
        reason = "invalid_request"
    elif isinstance(error, openai.APIStatusError):
        reason = ModelError.reason_from_status_code(status_code) or "provider_error"
    elif isinstance(error, openai.APIError):
        reason = ModelError.reason_from_status_code(status_code) or "provider_error"
    else:
        reason = ModelError.reason_from_status_code(status_code) or "unknown"

    return (
        reason,
        ModelError.is_reason_recoverable(reason),
        status_code,
        provider_code,
    )


def build_openai_model_error(error: Exception, model_id: str) -> ModelError:
    """Build a normalized ModelError from an OpenAI SDK exception."""
    reason, recoverable, status_code, provider_code = classify_openai_error(error)
    return ModelError(
        message=f"OpenAI error for {model_id}: {error}",
        model_id=model_id,
        reason=reason,
        recoverable=recoverable,
        status_code=status_code,
        provider_code=provider_code,
        details={"error_type": reason},
        cause=error,
    )
