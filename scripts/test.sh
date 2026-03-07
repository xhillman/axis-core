#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "error: expected project virtualenv at $VENV_PYTHON" >&2
    echo 'create it first, for example: python3 -m venv .venv && ./.venv/bin/pip install -e ".[dev]"' >&2
    exit 1
fi

# Keep local runs isolated from unrelated globally auto-loaded pytest plugins while still loading
# the plugins this repo expects from the project virtualenv.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

exec "$VENV_PYTHON" -m pytest -p pytest_asyncio.plugin -p pytest_cov "$@"
