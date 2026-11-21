#!/bin/sh
# Use the formal training/simulation environment, or an explicit override.
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
PYTHON_BIN=${SANDBOX_PYTHON:-"$PROJECT_ROOT/../environments/train/bin/python"}
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Missing formal Python environment: $PYTHON_BIN" >&2
    exit 1
fi
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" "$@"
