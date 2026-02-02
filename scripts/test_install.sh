#!/bin/bash
# Test that a published version installs correctly
# Usage: ./scripts/test_install.sh 0.1.3 [testpypi|pypi]

set -e

VERSION=$1
SOURCE=${2:-testpypi}

if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version> [testpypi|pypi]"
    echo "Example: $0 0.1.3 testpypi"
    exit 1
fi

# Create temporary test directory
TEST_DIR=$(mktemp -d)
echo "Testing in: $TEST_DIR"
cd "$TEST_DIR"

# Create venv
echo "Creating fresh virtual environment..."
python3 -m venv .venv

# Install package
echo "Installing axis-core==$VERSION from $SOURCE..."
if [ "$SOURCE" = "pypi" ]; then
    .venv/bin/pip install "axis-core[anthropic]==$VERSION"
else
    # Use pip (not uv) for TestPyPI - uv has compatibility issues
    .venv/bin/pip install \
      --index-url https://test.pypi.org/simple/ \
      --extra-index-url https://pypi.org/simple/ \
      "axis-core[anthropic]==$VERSION"
fi

# Run verification tests
echo ""
echo "Running verification tests..."

.venv/bin/python << 'PYTHON'
import sys

# Test 1: Version check
import axis_core
print(f"✓ Version: {axis_core.__version__}")

# Test 2: Sentinel exists (config defaults feature)
from axis_core import agent as agent_module
has_sentinel = hasattr(agent_module, '_UNSET')
print(f"✓ Has _UNSET sentinel: {has_sentinel}")
if not has_sentinel:
    print("✗ Missing config defaults feature!")
    sys.exit(1)

# Test 3: Lazy registration works
from axis_core.engine.registry import model_registry, memory_registry, planner_registry
models = model_registry.list()
print(f"✓ Models registered: {len(models)} ({', '.join(sorted(models)[:3])}...)")
print(f"✓ Memory adapters: {memory_registry.list()}")
print(f"✓ Planners: {planner_registry.list()}")

if len(models) < 10:
    print("✗ Expected at least 10 models registered!")
    sys.exit(1)

# Test 4: Config defaults work
# The config singleton is already initialized, so we modify it directly
from axis_core.config import config
original_model = config.default_model

config.default_model = 'test-model'
config.default_planner = 'test-planner'

from axis_core import Agent
agent = Agent()
if agent._model != 'test-model':
    print(f"✗ Config defaults not working! Got: {agent._model}")
    sys.exit(1)
print(f"✓ Config defaults working")

# Restore original
config.default_model = original_model

# Test 5: Basic import and creation
try:
    from axis_core import Agent, tool, Budget
    print("✓ Core imports working")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

print("\n✅ All tests passed!")
PYTHON

echo ""
echo "✅ Installation of axis-core==$VERSION from $SOURCE verified!"
echo ""
echo "Cleanup: rm -rf $TEST_DIR"
