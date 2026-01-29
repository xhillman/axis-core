"""Telemetry adapters for axis-core.

This module provides implementations of the TelemetrySink protocol for various
observability backends.
"""

from __future__ import annotations


def __getattr__(name: str):
    """Lazy loading of telemetry sinks to avoid circular imports."""
    if name == "ConsoleSink":
        from axis_core.adapters.telemetry.console import ConsoleSink

        return ConsoleSink

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ConsoleSink"]
