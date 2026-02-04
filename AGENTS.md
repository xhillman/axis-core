# Repository Guidelines

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

## Commit & Pull Request Guidelines

- Use short, imperative, lowercase commit subjects (example: `add ReAct planner`).
- PRs should include a brief summary, rationale, and the exact test command(s) run.
- Link related issues when available and update `README.md` for user-facing changes.
- If dependencies change, regenerate `requirements.lock` and note it in the PR.

## Security & Configuration Tips

- Environment variables are documented in `.env.example`; never commit secrets.
- Keep `requirements.lock` current to support supply-chain security checks.

## Architecture Overview

- The runtime follows a three-layer design: Agent API, Lifecycle Engine, and pluggable Adapters.
- Adapters are resolved lazily and registered via registries in `axis_core/engine/`.

## Documentation

README.md should be kept up to date to reflect the current state of the project. A section that includes future enhancements and planned features is encouraged to give users insight into the roadmap.

After completing significant features or changes, ensure that the README.md is updated accordingly.

## Project Authority Map

- Task List: tasks-axis-core-prd.md (source of truth for task IDs, sub-tasks, file targets)
- Architecture & Constraints: SPEC.md (ADRs, invariants, design rules)
- Execution Process: process-tasks.md (TDD flow, pacing, completion protocol)
- Product Intent (only if needed): axis-core-prd.md
