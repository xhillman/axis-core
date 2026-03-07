"""Shared provider model catalog metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class ModelCatalogEntry:
    """Catalog metadata for a registered provider model."""

    model_id: str
    aliases: tuple[str, ...] = ()
    uses_completion_tokens: bool = False
    uses_responses_api: bool = False


ANTHROPIC_MODEL_CATALOG: Final[tuple[ModelCatalogEntry, ...]] = (
    ModelCatalogEntry(model_id="claude-3-haiku-20240307"),
    ModelCatalogEntry(model_id="claude-sonnet-4-20250514"),
    ModelCatalogEntry(model_id="claude-opus-4-20250514"),
    ModelCatalogEntry(model_id="claude-opus-4-1-20250805"),
    ModelCatalogEntry(model_id="claude-sonnet-4-5-20250929", aliases=("claude-sonnet",)),
    ModelCatalogEntry(model_id="claude-haiku-4-5-20251001", aliases=("claude-haiku",)),
    ModelCatalogEntry(model_id="claude-opus-4-5-20251101"),
    ModelCatalogEntry(model_id="claude-opus-4-6", aliases=("claude-opus",)),
)


OPENAI_MODEL_CATALOG: Final[tuple[ModelCatalogEntry, ...]] = (
    ModelCatalogEntry(model_id="gpt-5.2", uses_completion_tokens=True),
    ModelCatalogEntry(model_id="gpt-5.1", uses_completion_tokens=True),
    ModelCatalogEntry(model_id="gpt-5", uses_completion_tokens=True),
    ModelCatalogEntry(model_id="gpt-5-mini", uses_completion_tokens=True),
    ModelCatalogEntry(model_id="gpt-5-nano", uses_completion_tokens=True),
    ModelCatalogEntry(model_id="gpt-5.2-chat-latest", uses_completion_tokens=True),
    ModelCatalogEntry(model_id="gpt-5.1-chat-latest", uses_completion_tokens=True),
    ModelCatalogEntry(model_id="gpt-5-chat-latest", uses_completion_tokens=True),
    ModelCatalogEntry(model_id="gpt-5.2-pro", uses_completion_tokens=True),
    ModelCatalogEntry(model_id="gpt-5-pro", uses_completion_tokens=True),
    ModelCatalogEntry(model_id="gpt-4.1", uses_completion_tokens=True),
    ModelCatalogEntry(model_id="gpt-4.1-mini", uses_completion_tokens=True),
    ModelCatalogEntry(model_id="gpt-4.1-nano", uses_completion_tokens=True),
    ModelCatalogEntry(model_id="gpt-4o"),
    ModelCatalogEntry(model_id="gpt-4o-2024-05-13"),
    ModelCatalogEntry(model_id="gpt-4o-mini"),
    ModelCatalogEntry(model_id="o1", uses_completion_tokens=True),
    ModelCatalogEntry(model_id="o1-pro", uses_completion_tokens=True),
    ModelCatalogEntry(model_id="o1-mini", uses_completion_tokens=True),
    ModelCatalogEntry(model_id="o3", uses_completion_tokens=True),
    ModelCatalogEntry(model_id="o3-pro", uses_completion_tokens=True),
    ModelCatalogEntry(model_id="o3-mini", uses_completion_tokens=True),
    ModelCatalogEntry(model_id="o4-mini", uses_completion_tokens=True),
    ModelCatalogEntry(
        model_id="gpt-5.2-codex",
        uses_completion_tokens=True,
        uses_responses_api=True,
    ),
    ModelCatalogEntry(
        model_id="gpt-5.1-codex-max",
        uses_completion_tokens=True,
        uses_responses_api=True,
    ),
    ModelCatalogEntry(
        model_id="gpt-5.1-codex",
        uses_completion_tokens=True,
        uses_responses_api=True,
    ),
    ModelCatalogEntry(
        model_id="gpt-5-codex",
        uses_completion_tokens=True,
        uses_responses_api=True,
    ),
    ModelCatalogEntry(
        model_id="gpt-5.1-codex-mini",
        uses_completion_tokens=True,
        uses_responses_api=True,
    ),
    ModelCatalogEntry(
        model_id="codex-mini-latest",
        uses_completion_tokens=True,
        uses_responses_api=True,
    ),
    ModelCatalogEntry(model_id="gpt-5-search", uses_responses_api=True),
    ModelCatalogEntry(
        model_id="gpt-5-search-api",
        uses_completion_tokens=True,
        uses_responses_api=True,
    ),
    ModelCatalogEntry(model_id="gpt-4o-search-preview", uses_responses_api=True),
    ModelCatalogEntry(model_id="gpt-4o-mini-search-preview", uses_responses_api=True),
    ModelCatalogEntry(
        model_id="o3-deep-research",
        uses_completion_tokens=True,
        uses_responses_api=True,
    ),
    ModelCatalogEntry(
        model_id="o4-mini-deep-research",
        uses_completion_tokens=True,
        uses_responses_api=True,
    ),
    ModelCatalogEntry(model_id="computer-use-preview", uses_responses_api=True),
)


def iter_registered_model_targets(
    catalog: tuple[ModelCatalogEntry, ...],
) -> tuple[tuple[str, str], ...]:
    """Return `(registered_name, target_model_id)` pairs for catalog registration."""
    registrations: list[tuple[str, str]] = []
    for entry in catalog:
        registrations.append((entry.model_id, entry.model_id))
        registrations.extend((alias, entry.model_id) for alias in entry.aliases)
    return tuple(registrations)


def _build_alias_targets(catalog: tuple[ModelCatalogEntry, ...]) -> dict[str, str]:
    alias_targets: dict[str, str] = {}
    for entry in catalog:
        for alias in entry.aliases:
            alias_targets[alias] = entry.model_id
    return alias_targets


def _build_completion_token_model_ids(
    catalog: tuple[ModelCatalogEntry, ...],
) -> frozenset[str]:
    return frozenset(entry.model_id for entry in catalog if entry.uses_completion_tokens)


def _build_responses_api_model_ids(
    catalog: tuple[ModelCatalogEntry, ...],
) -> frozenset[str]:
    return frozenset(entry.model_id for entry in catalog if entry.uses_responses_api)


ANTHROPIC_ALIAS_TARGETS: Final[dict[str, str]] = _build_alias_targets(ANTHROPIC_MODEL_CATALOG)
OPENAI_COMPLETION_TOKENS_MODEL_IDS: Final[frozenset[str]] = _build_completion_token_model_ids(
    OPENAI_MODEL_CATALOG
)
OPENAI_RESPONSES_API_MODEL_IDS: Final[frozenset[str]] = _build_responses_api_model_ids(
    OPENAI_MODEL_CATALOG
)

