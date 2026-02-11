# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- Added lifecycle policy enforcement tests covering phase timeouts, retry exhaustion, rate-limit
  breach behavior, model/tool cache hits, and non-cacheable bypass.
- Added `Agent(confirmation_handler=...)` and chainable `agent.on_confirm(...)` APIs for
  destructive tool approvals.
- Added lifecycle tests for destructive-tool confirmation approval, rejection, missing-handler,
  and malformed callback return paths.
- Added versioned checkpoint envelopes (`version`, `phase`, `next_phase`, `saved_at`, serialized
  `context`) built from `RunContext.serialize()`.
- Added public `Agent.resume()` / `Agent.resume_async()` APIs for checkpoint-based run resumption.
- Added `OpenAIResponsesModel` adapter for OpenAI Responses API payloads (`responses.create`) with
  normalization into existing `ModelResponse`/`ModelChunk` contracts.
- Added unit coverage for OpenAI Responses routing/mapping and a gated live integration test
  (`OPENAI_API_KEY`) for Responses-backed model execution.
- Added GitHub Actions quality-gate workflow for PR/release paths with dedicated `pytest`,
  `ruff`, and `mypy` jobs.

### Changed

- Engine now enforces per-phase timeouts (`observe`, `plan`, `act`, `evaluate`, `finalize`) in
  addition to wall-time budget limits.
- Added runtime retry enforcement for model and tool steps using configured/global policy and
  step/tool overrides.
- Added runtime rate-limit enforcement for model calls and tool calls (global + tool-specific)
  using `RateLimiter`.
- Added in-memory TTL cache enforcement for model responses and cacheable tool results with
  max-size eviction.
- `Agent` now passes resolved runtime config into `LifecycleEngine.execute()` so timeout/retry/
  rate-limit/cache policies are applied consistently for `run*` and `stream*`.
- Act-phase tool execution now enforces confirmation before any `Capability.DESTRUCTIVE` tool
  runs, with deterministic errors for missing/rejected/malformed confirmation behavior.
- Lifecycle execution now persists deterministic checkpoints at phase boundaries
  (`initialize`, `observe`, `plan`, `act`, `evaluate`) and can resume from valid boundaries with
  explicit validation for corrupt/incompatible checkpoint state.
- Optional provider scope now replaces `ollama` with `openrouter` extras/docs and documents
  OpenRouter usage via the existing OpenAI-compatible adapter path.
- `OpenAIModel` now routes by model ID to Chat Completions or Responses API backends transparently,
  preserving the existing user-facing model adapter surface.
- Registered OpenAI Responses model IDs for codex/search/deep-research/computer-use under the
  existing `[openai]` optional extra and documented usage in README.
- Aligned beta-facing docs to shipped behavior: removed checkpoint/resume from planned roadmap
  items, refreshed PRD implementation-status notes, clarified OpenRouter as OpenAI-compatible
  routing (no dedicated `OpenRouterModel` class), and clarified memory string resolution as
  adapter-name based (`ephemeral`/`sqlite`/`redis`).

### Fixed

- Invalid `AXIS_MAX_CYCLE_CONTEXT` values no longer crash model-step execution; runtime now falls
  back to a safe default context window.
- Repeated per-step warnings for non-decorated tools (missing `_axis_manifest`) were reduced to a
  single warning per tool per engine instance.
- Security redaction now covers hyphenated secret-key variants (for example `x-api-key`) and
  free-form error strings before telemetry/checkpoint serialization.
- Act-phase failure telemetry and persisted `RunState` error/tool-call records now sanitize
  secret-like values in messages and exception causes.
- AutoPlanner now hardens malformed/partial JSON extraction and uses deterministic fallback-reason
  codes while validating tool-step schema mismatches (`tool` name and `args` object shape) before
  returning auto plans.

## [0.6.0] - 2026-02-08

### Added

- Added `FileSink` telemetry adapter with JSONL output and configurable buffering.
- Added `CallbackSink` telemetry adapter for user-supplied sync or async handlers.
- Added telemetry adapter tests for file and callback sinks.

### Changed

- Wired `AXIS_TELEMETRY_SINK=file` and `AXIS_TELEMETRY_SINK=callback` into `Agent`
  environment-based sink resolution.

## [0.5.1] - 2026-02-08

### Fixed

- Restored built-in adapter auto-registration during normal import paths (`from axis_core import Agent`).
- Fixed registry initialization order that could leave model/memory/planner registries empty and raise:
  `ConfigError: Unknown adapter 'claude-haiku'`.
- Added regression coverage to validate built-in registrations in a fresh interpreter without manual adapter imports.

## [0.5.0] - 2026-02-07

### Added

- Shared redaction utility and default redaction across telemetry, traces, and persisted run state.
- Real Anthropic integration test (slow, API-key gated) for end-to-end `RunResult` validation.
- Repository map and focused subsystem maps for low-context task routing.
- Explicit roadmap labeling in README for committed vs exploratory features.
- CHANGELOG/version bump policy documentation in CLAUDE.md and AGENTS.md.

### Changed

- Lifecycle engine split into per-phase modules under `axis_core/engine/phases/`.
- Timeout and wall-clock budget enforcement now behaves as hard runtime limits.
- Public package `config` export semantics clarified and validated.
- OpenAI model registrations trimmed to validated model IDs.
- Test suite hardened to prefer public-contract surfaces over private internals.

### Fixed

- Security hardening to prevent sensitive payload leakage in telemetry/trace/state.
- Misleading `auth` Agent API surface removed/deprecated to avoid false credential isolation assumptions.
- Attachment MIME validation tests and SQLite optional-dependency test import behavior.
- Real-provider integration test now skips on transient connection failures.

## [0.4.0] - 2026-02-06

### Added

- SQLiteMemory adapter with FTS5 keyword search and session support (`axis_core/adapters/memory/sqlite.py`).
- RedisMemory adapter with TTL, namespace support, and session persistence (`axis_core/adapters/memory/redis.py`).
- Lazy factory registration for SQLite and Redis memory adapters in the adapter registry.
- Tests for SQLiteMemory (20 tests) and RedisMemory (22 tests).

## [0.3.1] - 2026-02-06

### Added

- Attachment types (Image, PDF) with size limits and metadata serialization.
- CancelToken for cooperative cancellation.
- Attachment and cancellation wiring through Agent, RunContext, and Lifecycle.
- Tests covering attachments, cancellation, and lifecycle cancellation handling.

## [0.3.0] - 2026-02-05

### Added

- Session dataclass with message history, versioning, and serialization.
- SessionStore protocol for session persistence backends.
- `agent.session()` API for creating and resuming multi-turn conversations.
- Optimistic concurrency checks with ConcurrencyError on version conflicts.
- Session tests for create, resume, and version conflict handling.

## [0.2.0] - 2026-02-04

### Added

- True streaming functionality (`stream()` and `stream_async()` methods).
- ReAct planner with explicit reasoning steps.
- AutoPlanner with LLM-based tool selection and ordering.
- Model fallback system with automatic retry on recoverable errors.
- OpenAI model adapter (GPT-4, GPT-4o, o1/o3/o4 series).

### Fixed

- Run state overwrite bug in multi-cycle execution.
- Default config resolution for environment variables.

## [0.1.3] - 2026-02-02

### Added

- Auto-registration of adapters via lazy loading factories.
- Release automation scripts and checklist.

### Fixed

- Package testing script.

## [0.1.0] - 2026-01-30

### Added

- Initial project skeleton.
- Lifecycle engine with Observe → Plan → Act → Evaluate cycle.
- Agent API with `run()` and `run_async()` methods.
- `@tool` decorator with automatic JSON schema generation.
- Anthropic model adapter (Claude Opus, Sonnet, Haiku).
- EphemeralMemory adapter with keyword search.
- SequentialPlanner adapter.
- Console telemetry sink.
- Budget tracking (cycles, tokens, cost, wall time).
- Configuration system with environment variable support.
- Comprehensive error taxonomy with recovery classification.
- Adapter registry with entry point discovery.
