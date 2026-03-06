"""Transcript normalization and context-window helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContextWindowAssessment:
    """Result of evaluating an estimated prompt size against context thresholds."""

    estimated_tokens: int
    context_window_tokens: int
    remaining_tokens: int
    should_warn: bool
    should_block: bool


@dataclass(frozen=True)
class ContextWindowGuard:
    """Token-threshold guard evaluated before model calls."""

    warn_threshold_tokens: int | None = None
    block_threshold_tokens: int | None = None

    @staticmethod
    def _normalized_threshold(value: int | None) -> int | None:
        if value is None:
            return None
        return value if value > 0 else None

    def evaluate(
        self,
        *,
        estimated_tokens: int,
        context_window_tokens: int,
    ) -> ContextWindowAssessment:
        """Evaluate warn/block thresholds from an estimated token count."""
        normalized_estimated = max(0, estimated_tokens)
        normalized_window = max(0, context_window_tokens)
        remaining_tokens = max(0, normalized_window - normalized_estimated)

        block_threshold = self._normalized_threshold(self.block_threshold_tokens)
        warn_threshold = self._normalized_threshold(self.warn_threshold_tokens)

        should_block = (
            block_threshold is not None
            and remaining_tokens <= block_threshold
        )
        should_warn = (
            not should_block
            and warn_threshold is not None
            and remaining_tokens <= warn_threshold
        )

        return ContextWindowAssessment(
            estimated_tokens=normalized_estimated,
            context_window_tokens=normalized_window,
            remaining_tokens=remaining_tokens,
            should_warn=should_warn,
            should_block=should_block,
        )


def _is_empty_assistant_content(content: Any) -> bool:
    """Return True when assistant message content is effectively empty."""
    if content is None:
        return True
    if isinstance(content, str):
        return content == ""
    if isinstance(content, (list, tuple)):
        return len(content) == 0
    return False


def _normalize_tool_result_content(content: Any, max_tool_result_chars: int | None) -> str:
    """Normalize tool-result content and optionally cap very large payloads."""
    text = content if isinstance(content, str) else str(content)
    if max_tool_result_chars is None or max_tool_result_chars <= 0:
        return text
    if len(text) <= max_tool_result_chars:
        return text
    truncated_count = len(text) - max_tool_result_chars
    return f"{text[:max_tool_result_chars]}...[truncated {truncated_count} chars]"


def normalize_transcript_messages(
    messages: list[dict[str, Any]],
    *,
    strict: bool = False,
    max_tool_result_chars: int | None = None,
) -> list[dict[str, Any]]:
    """Repair transcript ordering/pairing for assistant tool-calls and tool results.

    Repairs out-of-order tool results by moving them after the matching assistant
    tool-call message, drops orphaned/duplicate tool results, and optionally
    rejects unresolved pairings when strict mode is enabled.
    """
    normalized: list[dict[str, Any]] = []
    deferred_tool_results: dict[str, list[dict[str, Any]]] = {}
    assistant_index_by_call_id: dict[str, int] = {}
    unresolved_call_ids: set[str] = set()
    matched_call_ids: set[str] = set()
    orphan_tool_results = 0
    duplicate_tool_results = 0

    for raw_msg in messages:
        if not isinstance(raw_msg, dict):
            continue

        msg = dict(raw_msg)
        role = msg.get("role")

        if role == "assistant":
            raw_tool_calls = msg.get("tool_calls")
            normalized_tool_calls: list[dict[str, Any]] = []
            if isinstance(raw_tool_calls, list):
                for raw_tool_call in raw_tool_calls:
                    if not isinstance(raw_tool_call, dict):
                        continue
                    call_id_raw = raw_tool_call.get("id")
                    if call_id_raw is None:
                        continue
                    call_id = str(call_id_raw).strip()
                    if not call_id:
                        continue
                    tool_call = dict(raw_tool_call)
                    tool_call["id"] = call_id
                    normalized_tool_calls.append(tool_call)

            if normalized_tool_calls:
                msg["tool_calls"] = normalized_tool_calls
            else:
                msg.pop("tool_calls", None)

            normalized.append(msg)
            assistant_idx = len(normalized) - 1

            for tool_call in normalized_tool_calls:
                call_id = str(tool_call["id"])
                unresolved_call_ids.add(call_id)
                assistant_index_by_call_id[call_id] = assistant_idx

                deferred_results = deferred_tool_results.get(call_id)
                if not deferred_results or call_id in matched_call_ids:
                    continue

                normalized.append(deferred_results.pop(0))
                matched_call_ids.add(call_id)
                unresolved_call_ids.discard(call_id)

                if deferred_results:
                    duplicate_tool_results += len(deferred_results)
                    deferred_results.clear()

        elif role == "tool":
            call_id_raw = msg.get("tool_call_id")
            if call_id_raw is None:
                orphan_tool_results += 1
                continue

            call_id = str(call_id_raw).strip()
            if not call_id:
                orphan_tool_results += 1
                continue

            tool_result_msg = dict(msg)
            tool_result_msg["role"] = "tool"
            tool_result_msg["tool_call_id"] = call_id
            tool_result_msg["content"] = _normalize_tool_result_content(
                msg.get("content", ""),
                max_tool_result_chars=max_tool_result_chars,
            )

            if call_id in matched_call_ids:
                duplicate_tool_results += 1
                continue

            if call_id in unresolved_call_ids:
                normalized.append(tool_result_msg)
                matched_call_ids.add(call_id)
                unresolved_call_ids.discard(call_id)
                continue

            deferred_tool_results.setdefault(call_id, []).append(tool_result_msg)

        else:
            normalized.append(msg)

    unresolved_orphan_ids = sorted(
        call_id
        for call_id, pending_results in deferred_tool_results.items()
        if pending_results
    )
    orphan_tool_results += sum(
        len(pending_results)
        for pending_results in deferred_tool_results.values()
    )

    if strict and (unresolved_call_ids or orphan_tool_results or duplicate_tool_results):
        issue_parts: list[str] = []
        if unresolved_call_ids:
            issue_parts.append(f"unresolved tool_calls={sorted(unresolved_call_ids)}")
        if unresolved_orphan_ids:
            issue_parts.append(f"orphan tool_results={unresolved_orphan_ids}")
        if duplicate_tool_results:
            issue_parts.append(f"duplicate tool_results={duplicate_tool_results}")
        raise ValueError(
            "Transcript integrity validation failed: " + "; ".join(issue_parts)
        )

    for call_id in sorted(unresolved_call_ids):
        assistant_idx_opt = assistant_index_by_call_id.get(call_id)
        if assistant_idx_opt is None:
            continue
        if assistant_idx_opt >= len(normalized):
            continue

        assistant_msg = normalized[assistant_idx_opt]
        tool_calls = assistant_msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue

        filtered_tool_calls = [
            tool_call
            for tool_call in tool_calls
            if str(tool_call.get("id", "")).strip() != call_id
        ]
        if filtered_tool_calls:
            assistant_msg["tool_calls"] = filtered_tool_calls
        else:
            assistant_msg.pop("tool_calls", None)

    cleaned_messages: list[dict[str, Any]] = []
    for msg in normalized:
        if msg.get("role") == "assistant":
            tool_calls = msg.get("tool_calls")
            has_tool_calls = isinstance(tool_calls, list) and len(tool_calls) > 0
            if not has_tool_calls and _is_empty_assistant_content(msg.get("content")):
                continue
        cleaned_messages.append(msg)

    return cleaned_messages


def _message_text_for_token_estimation(message: dict[str, Any]) -> str:
    """Build a stable text representation for heuristic token estimation."""
    role = str(message.get("role", ""))

    content = message.get("content", "")
    if isinstance(content, list):
        content_text = " ".join(str(part) for part in content)
    else:
        content_text = str(content)

    parts = [role, content_text]

    tool_call_id = message.get("tool_call_id")
    if tool_call_id is not None:
        parts.append(str(tool_call_id))

    raw_tool_calls = message.get("tool_calls")
    if isinstance(raw_tool_calls, list):
        for call in raw_tool_calls:
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("id", ""))
            call_name = str(call.get("name", ""))
            arguments = call.get("arguments", {})
            try:
                arguments_json = json.dumps(arguments, sort_keys=True, default=str)
            except (TypeError, ValueError):
                arguments_json = str(arguments)
            parts.append(f"{call_id}:{call_name}:{arguments_json}")

    return " ".join(parts)


def estimate_transcript_tokens(
    messages: list[dict[str, Any]],
    *,
    system: str | None = None,
) -> int:
    """Heuristically estimate transcript tokens using a character-based approximation."""
    text_parts: list[str] = []
    if system:
        text_parts.append(system)

    for message in messages:
        if isinstance(message, dict):
            text_parts.append(_message_text_for_token_estimation(message))

    transcript_text = "\n".join(text_parts)
    if not transcript_text:
        return 0
    return max(1, len(transcript_text) // 4)


def _first_tool_result_index(messages: list[dict[str, Any]]) -> int | None:
    for index, message in enumerate(messages):
        if message.get("role") == "tool":
            return index
    return None


def _first_prunable_non_tool_index(messages: list[dict[str, Any]]) -> int | None:
    latest_user_index = None
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            latest_user_index = index

    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "tool" or role == "system":
            continue
        if latest_user_index is not None and index == latest_user_index:
            continue
        return index
    return None


def prune_messages_for_context_window(
    messages: list[dict[str, Any]],
    *,
    target_tokens: int,
) -> tuple[list[dict[str, Any]], int]:
    """Prune transcript toward token target, preferring old tool results first."""
    normalized_target = max(1, target_tokens)
    working = normalize_transcript_messages(list(messages), strict=False)
    dropped = 0
    max_iterations = len(working)

    while (
        working
        and dropped < max_iterations
        and estimate_transcript_tokens(working) > normalized_target
    ):
        prune_index = _first_tool_result_index(working)
        if prune_index is None:
            prune_index = _first_prunable_non_tool_index(working)
        if prune_index is None:
            break

        del working[prune_index]
        dropped += 1
        working = normalize_transcript_messages(working, strict=False)

    return (working, dropped)
