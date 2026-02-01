# Release Scripts

Automation scripts for releasing axis-core packages.

## Scripts

### `bump_version.sh`
Bump version across all files consistently.

```bash
./scripts/bump_version.sh 0.1.3
```

Updates:
- `pyproject.toml`
- `axis_core/__init__.py`

### `publish.sh`
Build, verify, and publish package with safety checks.

```bash
# Publish to TestPyPI (recommended first)
./scripts/publish.sh testpypi

# Publish to production PyPI
./scripts/publish.sh pypi
```

Checks:
- ✓ Version consistency
- ✓ Tests passing
- ✓ Wheel contents
- ✓ Safety prompt for PyPI

### `test_install.sh`
Verify a published package installs correctly.

```bash
# Test TestPyPI version
./scripts/test_install.sh 0.1.3 testpypi

# Test PyPI version
./scripts/test_install.sh 0.1.3 pypi
```

Tests:
- ✓ Correct version
- ✓ Config defaults feature
- ✓ Lazy registration working
- ✓ All imports work

## Typical Release Workflow

```bash
# 1. Bump version
./scripts/bump_version.sh 0.1.3

# 2. Commit
git add -u
git commit -m "Bump version to 0.1.3"

# 3. Publish to TestPyPI
./scripts/publish.sh testpypi

# 4. Test installation
./scripts/test_install.sh 0.1.3 testpypi

# 5. If tests pass, publish to PyPI
./scripts/publish.sh pypi

# 6. Tag release
git tag -a v0.1.3 -m "Release v0.1.3"
git push origin main --tags
```

## See Also

- [RELEASING.md](../RELEASING.md) - Detailed release documentation
- [pyproject.toml](../pyproject.toml) - Package metadata
