#!/bin/bash
# Safe publish script with verification
# Usage: ./scripts/publish.sh [testpypi|pypi]

set -e

TARGET=${1:-testpypi}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Axis-Core Publish Script ===${NC}"
echo ""

# 1. Check versions match
PYPROJECT_VERSION=$(grep "^version = " pyproject.toml | cut -d'"' -f2)
INIT_VERSION=$(grep "__version__ = " axis_core/__init__.py | head -1 | cut -d'"' -f2)

echo "Checking version consistency..."
echo "  pyproject.toml: $PYPROJECT_VERSION"
echo "  __init__.py:    $INIT_VERSION"

if [ "$PYPROJECT_VERSION" != "$INIT_VERSION" ]; then
    echo -e "${RED}✗ Version mismatch!${NC}"
    echo "  Run: ./scripts/bump_version.sh $PYPROJECT_VERSION"
    exit 1
fi
echo -e "${GREEN}✓ Versions match${NC}"
echo ""

# 2. Run tests
echo "Running test suite..."
if ! pytest tests/ -q; then
    echo -e "${RED}✗ Tests failed!${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Tests passed${NC}"
echo ""

# 3. Clean and build
echo "Building package..."
rm -rf dist/ build/ *.egg-info 2>/dev/null || true
uv build
echo -e "${GREEN}✓ Build complete${NC}"
echo ""

# 4. Verify wheel contents
echo "Verifying wheel contents..."
WHEEL=$(ls dist/*.whl)
if unzip -l "$WHEEL" | grep -q "axis_core/agent.py"; then
    echo -e "${GREEN}✓ Wheel contains axis_core/agent.py${NC}"
else
    echo -e "${RED}✗ Wheel missing expected files${NC}"
    exit 1
fi
echo ""

# 5. Show what will be uploaded
echo "Package to upload:"
ls -lh dist/
echo ""

# 6. Upload
if [ "$TARGET" = "pypi" ]; then
    echo -e "${YELLOW}⚠ Publishing to PRODUCTION PyPI${NC}"
    read -p "Are you sure? (yes/no): " -r
    if [[ ! $REPLY =~ ^yes$ ]]; then
        echo "Cancelled"
        exit 0
    fi
    uv publish
else
    echo -e "${GREEN}Publishing to TestPyPI${NC}"
    uv publish --publish-url https://test.pypi.org/legacy/
fi

echo ""
echo -e "${GREEN}✓ Published version $PYPROJECT_VERSION to $TARGET${NC}"
echo ""
echo "Installation instructions:"
if [ "$TARGET" = "pypi" ]; then
    echo "  pip install 'axis-core[anthropic]==$PYPROJECT_VERSION'"
else
    echo "  uv pip install --index-url https://test.pypi.org/simple/ \\"
    echo "                 --extra-index-url https://pypi.org/simple/ \\"
    echo "                 --index-strategy unsafe-best-match \\"
    echo "                 'axis-core[anthropic]==$PYPROJECT_VERSION'"
fi
