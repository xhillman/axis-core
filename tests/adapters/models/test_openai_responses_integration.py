"""Live OpenAI Responses API integration coverage."""

from __future__ import annotations

import os

import pytest

import axis_core.adapters.memory  # noqa: F401 - ensure built-in memory registration side effects
import axis_core.adapters.models  # noqa: F401 - ensure built-in model registration side effects
import axis_core.adapters.planners  # noqa: F401 - ensure planner registration side effects
from axis_core import Agent
from axis_core.result import RunResult, RunStats

pytest.importorskip("openai", reason="openai package not installed")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; skipping live OpenAI integration test",
)
async def test_real_openai_responses_run_returns_valid_run_result_structure() -> None:
    """A real OpenAI Responses model call should produce a successful RunResult."""
    agent = Agent(
        model="gpt-5-codex",
        planner="sequential",
        memory="ephemeral",
        system="Reply briefly and clearly.",
    )

    result = await agent.run_async(
        "Return exactly three words describing axis-core.",
        timeout=30.0,
    )

    if not result.success and result.error is not None:
        error_text = result.error.message.lower()
        if any(
            token in error_text
            for token in (
                "connection error",
                "model_not_found",
                "does not have access",
                "insufficient permissions",
            )
        ):
            pytest.skip(
                "Skipping due to environment-specific OpenAI Responses limitation: "
                f"{result.error.message}"
            )

    assert isinstance(result, RunResult)
    assert result.success is True
    assert result.error is None
    assert isinstance(result.output, str)
    assert result.output.strip() != ""
    assert isinstance(result.output_raw, str)
    assert result.output_raw.strip() != ""
    assert isinstance(result.stats, RunStats)
    assert result.stats.model_calls >= 1
    assert result.stats.total_tokens > 0
    assert result.run_id != ""
