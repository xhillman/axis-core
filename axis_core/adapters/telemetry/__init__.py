"""Telemetry adapters for axis-core.

This module provides implementations of the TelemetrySink protocol for various
observability backends.
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    """Lazy loading of telemetry sinks to avoid circular imports."""
    if name == "ConsoleSink":
        from axis_core.adapters.telemetry.console import ConsoleSink

        return ConsoleSink
    if name == "FileSink":
        from axis_core.adapters.telemetry.file import FileSink

        return FileSink
    if name == "CallbackSink":
        from axis_core.adapters.telemetry.callback import CallbackSink

        return CallbackSink

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ConsoleSink", "FileSink", "CallbackSink"]
