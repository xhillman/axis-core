# Build, CI & Release Map

> **When to open:** Changing packaging, publishing, versioning, or dependency management.

## Key Files

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, deps, optional extras, ruff/mypy config |
| `requirements.lock` | Pinned dependencies (all transitive) |
| `scripts/bump_version.sh` | Updates version in `pyproject.toml` + `__init__.py` |
| `scripts/publish.sh` | Build + upload to TestPyPI or PyPI |
| `scripts/test_install.sh` | Verify published package in fresh venv |
| `scripts/pypi_checklist.md` | Manual checklist for releases |
| `RELEASING.md` | Release process documentation |
| `CHANGELOG.md` | Version history |
| `dist/` | Built wheels + tarballs (`.gitignore` inside) |
| `pytest.ini` | Test runner configuration |

## Build System

- **Backend:** Hatchling (`hatchling.build`)
- **Package:** `axis_core` directory → `axis-core` on PyPI
- **Python:** >=3.10 (3.10, 3.11, 3.12, 3.13)

## Optional Dependencies (extras)

| Extra | Packages |
|---|---|
| `anthropic` | `anthropic>=0.18` |
| `openai` | `openai>=1.0` |
| `redis` | `redis>=5.0` |
| `sqlite` | `aiosqlite>=0.19` |
| `ollama` | `ollama>=0.1` |
| `full` | All of the above |
| `dev` | pytest, pytest-asyncio, pytest-cov, mypy, ruff, fakeredis |

## Release Workflow

1. `./scripts/bump_version.sh <new-version>` — updates pyproject.toml + __init__.py
2. Run tests: `pytest && ruff check axis_core --fix && mypy axis_core --strict`
3. `./scripts/publish.sh testpypi` — build + upload to TestPyPI
4. `./scripts/test_install.sh <version> testpypi` — verify in fresh venv
5. `./scripts/publish.sh pypi` — publish to production PyPI
6. Update `CHANGELOG.md` and `README.md`

## Lockfile Management

```bash
# Regenerate after changing pyproject.toml
uv pip compile pyproject.toml -o requirements.lock

# Verify lockfile
pytest tests/test_lockfile.py

# Audit for vulnerabilities
pip-audit -r requirements.lock
```

## Common Change Patterns

- **New dependency** → add to `pyproject.toml` → regenerate `requirements.lock` → run `test_lockfile.py`
- **New optional extra** → add to `[project.optional-dependencies]` + add to `full` extra
- **Version bump** → use `scripts/bump_version.sh`, never edit manually
- **Change linting rules** → update `[tool.ruff.lint]` in `pyproject.toml`

## Sharp Edges

- Version must be updated in TWO places: `pyproject.toml` and `axis_core/__init__.py` (bump script handles this)
- `dist/` contains built artifacts — has its own `.gitignore`
- Current version: `0.4.1`
