"""Tests for context-window guard and transcript pruning helpers."""

from __future__ import annotations

from axis_core.context import ContextWindowGuard, prune_messages_for_context_window


class TestContextWindowGuard:
    """Behavior tests for warn/block threshold evaluation."""

    def test_warn_threshold_triggers_without_blocking(self) -> None:
        guard = ContextWindowGuard(
            warn_threshold_tokens=32_000,
            block_threshold_tokens=16_000,
        )

        assessment = guard.evaluate(
            estimated_tokens=71_000,
            context_window_tokens=100_000,
        )

        assert assessment.remaining_tokens == 29_000
        assert assessment.should_warn is True
        assert assessment.should_block is False

    def test_block_threshold_takes_precedence(self) -> None:
        guard = ContextWindowGuard(
            warn_threshold_tokens=32_000,
            block_threshold_tokens=16_000,
        )

        assessment = guard.evaluate(
            estimated_tokens=90_000,
            context_window_tokens=100_000,
        )

        assert assessment.remaining_tokens == 10_000
        assert assessment.should_warn is False
        assert assessment.should_block is True


class TestContextWindowPruning:
    """Behavior tests for tool-result-first pruning and pairing repair."""

    def test_pruning_drops_old_tool_results_before_other_turns(self) -> None:
        messages = [
            {"role": "user", "content": "request"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "name": "lookup", "arguments": {"q": "x"}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "x" * 600},
            {"role": "user", "content": "follow-up"},
        ]

        pruned, dropped = prune_messages_for_context_window(messages, target_tokens=60)

        assert dropped > 0
        assert all(msg.get("role") != "tool" for msg in pruned)
        assert all(
            not msg.get("tool_calls")
            for msg in pruned
            if msg.get("role") == "assistant"
        )

    def test_pruning_preserves_latest_user_turn_when_trimming_non_tool_messages(self) -> None:
        messages = [
            {"role": "user", "content": "a" * 600},
            {"role": "assistant", "content": "b" * 500},
            {"role": "user", "content": "latest-user-turn"},
        ]

        pruned, dropped = prune_messages_for_context_window(messages, target_tokens=80)

        assert dropped > 0
        assert pruned[-1]["role"] == "user"
        assert pruned[-1]["content"] == "latest-user-turn"
