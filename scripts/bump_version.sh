#!/bin/bash
# Bump version across all files consistently
# Usage: ./scripts/bump_version.sh 0.1.3

set -e

NEW_VERSION=$1

if [ -z "$NEW_VERSION" ]; then
    echo "Usage: $0 <new-version>"
    echo "Example: $0 0.1.3"
    exit 1
fi

echo "Bumping version to $NEW_VERSION..."

# Update pyproject.toml
sed -i '' "s/^version = .*/version = \"$NEW_VERSION\"/" pyproject.toml

# Update __init__.py
sed -i '' "s/__version__ = .*/__version__ = \"$NEW_VERSION\"/" axis_core/__init__.py

echo "✓ Updated pyproject.toml"
echo "✓ Updated axis_core/__init__.py"

# Verify
echo ""
echo "Verification:"
grep "^version" pyproject.toml
grep "__version__" axis_core/__init__.py | head -1

echo ""
echo "Next steps:"
echo "1. Review changes: git diff"
echo "2. Commit: git add -u && git commit -m 'Bump version to $NEW_VERSION'"
echo "3. Build: rm -rf dist/ && uv build"
echo "4. Upload: uv publish --publish-url https://test.pypi.org/legacy/"
