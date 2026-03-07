# Contributing

Thanks for contributing to axis-core.

## Development Setup

```bash
pip install -e ".[dev]"

# Optional provider/memory extras as needed
pip install -e ".[anthropic,openai,redis,full]"
```

Run tests through `./scripts/test.sh ...`, not bare `pytest`. The wrapper uses the project
`.venv`, disables third-party pytest plugin autoload, and explicitly loads the repo's required
plugins.

## Quality Gates

Run these before opening a PR:

```bash
./scripts/test.sh
ruff check axis_core tests --fix
mypy axis_core --strict
```

Useful variants:

```bash
./scripts/test.sh -m "not slow"
./scripts/test.sh --cov=axis_core
```

## Testing Policy (Public Contracts)

Tests should target public APIs and documented extension points, not private helpers.

Allowed surfaces include:

- `Agent.run()`, `run_async()`, `stream()`, `stream_async()`
- `LifecycleEngine.execute()`
- Lifecycle phase methods (`_initialize`, `_observe`, `_plan`, `_act`, `_evaluate`, `_finalize`)
- Adapter protocol methods (`complete`, `stream`, `store`, `retrieve`, `plan`)
- `resolve_adapter()` and registry APIs

Avoid direct tests of internal private helpers that are not documented extension points.

## Coding Style

- Python `3.10+`
- Type hints required
- `mypy` strict mode
- `ruff` lint/format standards with 100-char line length
- `snake_case` for functions/variables, `PascalCase` for classes

## Pull Request Expectations

Include in each PR:

- Summary of behavior changes
- Rationale and risk notes
- Exact test commands run
- Related issue links (if applicable)

For user-facing changes, update in the same PR:

- `README.md`
- `CHANGELOG.md`

If dependencies change:

```bash
uv pip compile pyproject.toml -o requirements.lock
pip install -r requirements.lock
./scripts/test.sh tests/test_lockfile.py
```

## Security and Secrets

- Never commit secrets or real API keys.
- Use environment variables documented in `.env.example`.

## Project Layout

- Core package: `axis_core/`
- Tests: `tests/`
- Examples: `examples/`
- Specs/process docs: `dev/`
- Utility scripts: `scripts/`
