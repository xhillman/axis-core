"""Tests for tool idempotency helpers and ToolContext plumbing."""

from __future__ import annotations

import pytest

from axis_core.budget import Budget, BudgetState
from axis_core.tool import (
    ToolContext,
    build_idempotency_key,
    get_idempotent_result,
    run_idempotent,
    set_idempotent_result,
)


def _make_tool_context(*, idempotency_key: str | None) -> ToolContext:
    return ToolContext(
        run_id="run-1",
        agent_id="agent-1",
        cycle=0,
        context={},
        budget=Budget(),
        budget_state=BudgetState(),
        idempotency_key=idempotency_key,
    )


class TestIdempotencyHelpers:
    """Task 5 helper utility tests."""

    def test_build_idempotency_key_is_stable(self) -> None:
        key = build_idempotency_key(
            run_id="run-1",
            cycle=3,
            step_id="tool-step",
            tool_name="charge_card",
        )
        assert key == "axis:run-1:3:tool-step:charge_card"

    def test_get_set_idempotent_result_round_trip(self) -> None:
        ctx = _make_tool_context(idempotency_key="idem-1")

        found, _ = get_idempotent_result(ctx)
        assert found is False

        set_idempotent_result(ctx, {"status": "ok"})
        found, result = get_idempotent_result(ctx)
        assert found is True
        assert result == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_run_idempotent_dedupes_when_key_present(self) -> None:
        ctx = _make_tool_context(idempotency_key="idem-2")
        calls: dict[str, int] = {"count": 0}

        async def side_effect() -> str:
            calls["count"] += 1
            return f"effect-{calls['count']}"

        first = await run_idempotent(ctx, side_effect)
        second = await run_idempotent(ctx, side_effect)

        assert first == "effect-1"
        assert second == "effect-1"
        assert calls["count"] == 1

    @pytest.mark.asyncio
    async def test_run_idempotent_does_not_dedupe_without_key(self) -> None:
        ctx = _make_tool_context(idempotency_key=None)
        calls: dict[str, int] = {"count": 0}

        async def side_effect() -> str:
            calls["count"] += 1
            return f"effect-{calls['count']}"

        first = await run_idempotent(ctx, side_effect)
        second = await run_idempotent(ctx, side_effect)

        assert first == "effect-1"
        assert second == "effect-2"
        assert calls["count"] == 2

    def test_tool_context_idempotency_key_is_read_only(self) -> None:
        ctx = _make_tool_context(idempotency_key="idem-3")

        with pytest.raises(AttributeError, match="idempotency_key is read-only"):
            ctx.idempotency_key = "different"
