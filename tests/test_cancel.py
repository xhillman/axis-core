"""Tests for cancellation token."""

from __future__ import annotations

from axis_core.cancel import CancelToken


def test_cancel_token_defaults() -> None:
    """CancelToken should start not cancelled with no reason."""
    token = CancelToken()

    assert token.is_cancelled is False
    assert token.reason is None


def test_cancel_token_sets_reason() -> None:
    """CancelToken.cancel should set cancelled state and reason."""
    token = CancelToken()

    token.cancel("Stop")

    assert token.is_cancelled is True
    assert token.reason == "Stop"


def test_cancel_token_default_reason() -> None:
    """CancelToken.cancel should use default reason when none provided."""
    token = CancelToken()

    token.cancel()

    assert token.is_cancelled is True
    assert token.reason == "Cancelled by user"
