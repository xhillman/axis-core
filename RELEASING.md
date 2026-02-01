# Release Process

This document describes how to release a new version of axis-core.

## Quick Start

```bash
# 1. Bump version
./scripts/bump_version.sh 0.1.3

# 2. Build and publish to TestPyPI
./scripts/publish.sh testpypi

# 3. Test the package
cd ../test-project
uv pip install --upgrade \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  --index-strategy unsafe-best-match \
  'axis-core[anthropic]==0.1.3'

# 4. If tests pass, publish to PyPI
cd ../axis-core
./scripts/publish.sh pypi
```

## Detailed Steps

### 1. Version Bumping

Always bump version in both files:

- `pyproject.toml` - line 6: `version = "X.Y.Z"`
- `axis_core/__init__.py` - line 22: `__version__ = "X.Y.Z"`

Use the script to avoid mismatches:

```bash
./scripts/bump_version.sh 0.1.3
```

**Version scheme:**

- `0.1.x` - Alpha releases (breaking changes OK)
- `0.x.0` - Beta releases (API stabilizing)
- `1.0.0` - First stable release
- `1.x.y` - Patch (y) and minor (x) updates

### 2. Pre-Release Checklist

- [ ] All tests passing: `pytest tests/`
- [ ] Lint clean: `ruff check axis_core/`
- [ ] Type check clean: `mypy axis_core/ --strict`
- [ ] CHANGELOG.md updated
- [ ] Version bumped in both files
- [ ] Git committed: `git add -u && git commit -m "Release v0.1.3"`

### 3. Build Package

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build with uv (fast)
uv build

# Verify contents
ls -lh dist/
unzip -l dist/axis_core-*.whl | grep axis_core
```

### 4. Publish to TestPyPI

Always test on TestPyPI first!

```bash
uv publish --publish-url https://test.pypi.org/legacy/
```

Or use the script:

```bash
./scripts/publish.sh testpypi
```

### 5. Test Installation from TestPyPI

**Important:** Create a fresh venv to test:

```bash
# Create test project
mkdir -p ~/test-axis-core
cd ~/test-axis-core

# Create fresh venv
uv venv
source .venv/bin/activate

# Install from TestPyPI (specify exact version!)
uv pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  --index-strategy unsafe-best-match \
  'axis-core[anthropic]==0.1.3'

# Verify version
python -c "import axis_core; print(axis_core.__version__)"

# Test basic functionality
python -c "from axis_core import Agent; agent = Agent(); print('✓ Works!')"
```

### 6. Publish to Production PyPI

**WARNING:** Once published to PyPI, you CANNOT delete or modify that version!

```bash
uv publish

# Or with the script (includes safety prompt)
./scripts/publish.sh pypi
```

### 7. Tag Release

```bash
git tag -a v0.1.3 -m "Release v0.1.3"
git push origin v0.1.3
```

## Common Issues

### Issue: uv installs old version despite new upload

**Cause:** uv caches packages

**Solution:**

```bash
# Clear cache and reinstall
rm -rf .venv
uv venv
uv pip install --no-cache 'axis-core[anthropic]==0.1.3' ...
```

### Issue: Version mismatch between files

**Cause:** Manual version bumps in only one file

**Solution:**

```bash
# Use the script
./scripts/bump_version.sh 0.1.3

# Or verify manually
grep "^version" pyproject.toml
grep "__version__" axis_core/__init__.py
```

### Issue: "No solution found" when installing from TestPyPI

**Cause:** Missing `--index-strategy unsafe-best-match`

**Solution:**

```bash
# Always include this for TestPyPI
uv pip install \
  --index-strategy unsafe-best-match \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  'axis-core[anthropic]==X.Y.Z'
```

### Issue: TestPyPI version not found

**Cause:** Package name or version doesn't exist

**Solution:**

```bash
# Check what's on TestPyPI
curl -s https://test.pypi.org/pypi/axis-core/json | \
  python -c "import sys, json; print(list(json.load(sys.stdin)['releases'].keys()))"
```

## TestPyPI vs PyPI Differences

| Aspect | TestPyPI | PyPI |
| -------- | ---------- | ------ |
| Purpose | Testing uploads | Production |
| Data retention | May be deleted | Permanent |
| Version deletion | Sometimes possible | Never possible |
| Dependencies | Incomplete | Complete |
| Installation | Requires --index-strategy | Standard pip install |
| URL | test.pypi.org | pypi.org |

## Installation Examples

### From TestPyPI (for testing)

```bash
# uv (recommended)
uv pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  --index-strategy unsafe-best-match \
  'axis-core[anthropic]==0.1.3'

# pip (alternative)
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            'axis-core[anthropic]==0.1.3'
```

### From PyPI (production)

```bash
# uv (recommended)
uv pip install 'axis-core[anthropic]'

# pip (alternative)
pip install 'axis-core[anthropic]'
```

## Rollback

If you publish a broken version to PyPI, you **cannot delete it**. Instead:

1. Fix the issue
2. Bump to next patch version (0.1.3 → 0.1.4)
3. Publish the fix
4. (Optional) Yank the broken version on PyPI web UI

**Yanking** hides the version from `pip install axis-core` but allows `pip install axis-core==0.1.3` to still work.
