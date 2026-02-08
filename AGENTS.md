# Repository Guidelines

## Mandatory: Repo Map First

Before starting any task:

1. Open `REPO_MAP.md` at the repo root
2. Find your task type in the **Task Router** table
3. Open only the referenced `.agent/maps/*.md` file(s)
4. Work from those maps — avoid reading unrelated files to minimize context usage
5. If `REPO_MAP.md` is missing or outdated, regenerate it as the first step

For meta-process or prompt-document updates, route via `REPO_MAP.md` to `.agent/maps/meta_process.md`.

## Project Structure & Module Organization

- `axis_core/` is the core package. Key areas include `adapters/`, `engine/`, `loadouts/`, `protocols/`, and `testing/`.
- `tests/` holds the test suite and broadly mirrors the `axis_core/` layout.
- `examples/` contains usage demos.
- `dev/` includes design docs and specs.
- `scripts/` contains maintenance utilities.
- `requirements.lock` pins dependencies for reproducible installs.

## Build, Test, and Development Commands

- `pip install -e ".[dev]"` installs local dev dependencies.
- `pip install -e ".[anthropic,openai,redis,full]"` installs optional adapters.
- `pytest` runs the full test suite.
- `pytest --cov=axis_core` runs tests with coverage.
- `pytest -m "not slow"` skips slow tests.
- `ruff check axis_core tests --fix` lints and auto-fixes where safe.
- `mypy axis_core --strict` runs strict type checks.
- `uv pip compile pyproject.toml -o requirements.lock` regenerates the lockfile.
- `pip install -r requirements.lock` installs pinned dependencies.
- `pytest tests/test_lockfile.py` verifies the lockfile.

## Coding Style & Naming Conventions

- Python 3.10+ only.
- Type hints are required; `mypy` runs in strict mode.
- `ruff` enforces style with a 100-character line length.
- Use `snake_case` for functions/vars and `PascalCase` for classes.
- Tests use `test_*.py` files, `Test*` classes, and `test_*` functions.

## Testing Guidelines

- Frameworks: `pytest` + `pytest-asyncio` (asyncio mode auto).
- Mark tests with `@pytest.mark.unit`, `@pytest.mark.integration`, or `@pytest.mark.slow`.
- Async tests should use `@pytest.mark.asyncio`.
- Prefer absolute imports from `axis_core.*` in tests.

### Public-Contract Testing Policy

Tests must exercise behavior through **public API or documented extension points**, not private/internal methods. This prevents test brittleness when implementation details change.

**Allowed test surfaces:**

- `Agent.run()`, `run_async()`, `stream()`, `stream_async()` — agent-level public API
- `LifecycleEngine.execute()` — engine-level public API
- Lifecycle phase methods (`_initialize`, `_observe`, `_plan`, `_act`, `_evaluate`, `_finalize`) — documented architectural extension points
- Adapter protocol methods (`complete`, `stream`, `store`, `retrieve`, `plan`) — protocol contracts
- `resolve_adapter()` and registry APIs — public adapter resolution

**Do not test directly:**

- Internal helpers like `_try_models_with_fallback`, `_execute_model_step`, `_get_tool_manifests`, `_identify_exhausted_resource`
- Agent internals like `_build_engine`
- Any method prefixed with `_` that is not a lifecycle phase method

## Commit & Pull Request Guidelines

- Use short, imperative, lowercase commit subjects (example: `add ReAct planner`).
- PRs should include a brief summary, rationale, and the exact test command(s) run.
- Link related issues when available and update `README.md` for user-facing changes.
- If dependencies change, regenerate `requirements.lock` and note it in the PR.

## Security & Configuration Tips

- Environment variables are documented in `.env.example`; never commit secrets.
- Keep `requirements.lock` current to support supply-chain security checks.

## CHANGELOG & Versioning Policy

Every user-facing change should be reflected in `CHANGELOG.md` in the same PR.

### Version Bump Rules

- **Minor bump (`0.x.0` / `x.Y.0`)**: New public capabilities, new adapters/integrations, new public API surface, or behavior changes users must learn.
- **Patch bump (`0.0.x` / `x.y.Z`)**: Backward-compatible bug fixes, security hardening, performance improvements, and documentation clarifications without public API expansion.

### Changelog Entry Expectations

- Add an entry for: new features, bug fixes, security fixes, deprecations, and behavior changes.
- No entry required for: internal refactors/tests/chore work with no user-visible impact.

## Architecture Overview

- The runtime follows a three-layer design: Agent API, Lifecycle Engine, and pluggable Adapters.
- Adapters are resolved lazily and registered via registries in `axis_core/engine/`.

## Documentation

README.md should be kept up to date to reflect the current state of the project. A section that includes future enhancements and planned features is encouraged to give users insight into the roadmap.

After completing significant features or changes, ensure that the README.md is updated accordingly.

### README Currency Policy (Required)

For every PR that changes user-facing behavior, update `README.md` in the same PR.

Treat these as mandatory checks before marking work complete:

1. Update README sections affected by the change (API, config/env vars, adapters/models, limits, examples).
2. Ensure "Supported Models" and adapter capability statements match current registry/code behavior.
3. Ensure roadmap language uses the current "Planned (Committed)" vs "Exploring (Not Committed)" distinction.
4. If release/version messaging changes, keep README status/version references aligned with `CHANGELOG.md`.

## Project Authority Map

- Repo Router: `REPO_MAP.md` (entrypoint for minimal-context routing)
- Task List: tasks-axis-core-prd.md (source of truth for task IDs, sub-tasks, file targets)
- Architecture & Constraints: SPEC.md (ADRs, invariants, design rules)
- Execution Process: process-tasks.md (canonical source for TDD, quality gates, completion, failure handling)
- Assistant Memory: `dev/memory.md` (persistent user preferences + mistakes log)
- Task Summaries: `dev/task-summaries.md` (one concise summary per completed parent task)
- Production Safety Gate: `dev/production-safety-gate.md` (required before production release)
- Workflow Skills: `dev/skills/*/SKILL.md` (route, execute, gates, memory/summary, release safety)
- Product Intent (only if needed): axis-core-prd.md
- Execution Prompt Template: `dev/spec-driven.md` (behavioral guardrails; references process doc instead of duplicating mechanics)

## Process Ownership (Avoid Drift)

- Keep execution mechanics in `dev/process-tasks.md` only.
- Keep behavioral prompt constraints in `dev/spec-driven.md`.
- Keep routing guidance in `REPO_MAP.md` and `.agent/maps/*.md`.
- Keep persistent preferences/mistakes in `dev/memory.md` and update it as needed during execution.
- Keep completed-task summaries in `dev/task-summaries.md` and append entries after parent completion.
- Post the same concise parent-task summary in chat after completion.
- Run acceptance checks before parent execution and production safety checks before production release.
- When process policy changes, update `AGENTS.md`, `CLAUDE.md`, and related map docs in the same change.
- After policy-doc edits, run `python3 scripts/check_doc_policy_consistency.py`.
