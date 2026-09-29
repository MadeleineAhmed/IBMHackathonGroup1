#!/usr/bin/env bash
# run.sh — Execute a Python script (or command) inside the skore pyenv virtualenv
# without needing to activate anything first.
#
# Usage:
#   ./run.sh scripts/env_check.py        # environment sanity check
#   ./run.sh experiments/01_dummy.py     # run an experiment script
#   ./run.sh -m pytest tests -v          # run the test suite
#   ./run.sh -c "import skore; print(skore.__version__)"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prefer the team .venv (created by setup/unix/setup.sh); fall back to the
# 'skore' pyenv virtualenv.
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
else
    PYTHON="$(PYENV_VERSION=skore pyenv which python)"
fi

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 [-m module | -c command | script.py] [args...]"
    echo "       Runs the given Python script/command inside the project environment."
    exit 1
fi

exec "$PYTHON" "$@"
