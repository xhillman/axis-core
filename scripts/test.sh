#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "error: expected project virtualenv at $VENV_PYTHON" >&2
    echo 'create it first, for example: python3 -m venv .venv && ./.venv/bin/pip install -e ".[dev]"' >&2
    exit 1
fi

exec "$VENV_PYTHON" -m pytest "$@"
