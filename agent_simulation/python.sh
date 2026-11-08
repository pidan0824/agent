#!/bin/sh
# Run project commands with the configured Python environment and import paths.
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
PYTHON_BIN=${SANDBOX_PYTHON:-"$PROJECT_ROOT/.venv-memory/bin/python"}
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/.runtime${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" "$@"
