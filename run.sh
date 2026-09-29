#!/usr/bin/env bash
# run.sh — Execute a Python script (or command) inside the skore pyenv virtualenv
# without needing to activate anything first.
#
# Usage:
#   ./run.sh env_check.py            # run a script
#   ./run.sh test_skore.py           # run tests directly (bypasses pytest binary issue)
#   ./run.sh -m pytest test_skore.py # run pytest as a module (most reliable)
#   ./run.sh -c "import skore; print(skore.__version__)"

set -euo pipefail

PYENV_VERSION=skore
PYTHON="$(PYENV_VERSION=$PYENV_VERSION pyenv which python)"

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 [-m module | -c command | script.py] [args...]"
    echo "       Runs the given Python script/command inside the 'skore' virtualenv."
    exit 1
fi

exec "$PYTHON" "$@"
