"""Cancellation token for cooperative cancellation (AD-028)."""

from __future__ import annotations


class CancelToken:
    """Token for external cancellation."""

    def __init__(self) -> None:
        self._cancelled = False
        self._reason: str | None = None

    def cancel(self, reason: str = "Cancelled by user") -> None:
        """Request cancellation with optional reason."""
        self._cancelled = True
        self._reason = reason

    @property
    def is_cancelled(self) -> bool:
        """Whether cancellation has been requested."""
        return self._cancelled

    @property
    def reason(self) -> str | None:
        """Return the cancellation reason, if any."""
        return self._reason
