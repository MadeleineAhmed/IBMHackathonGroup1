#!/usr/bin/env bash
# One-time team setup for macOS / Linux.
#
# Usage (from anywhere):
#   bash setup/unix/setup.sh            # full setup
#   bash setup/unix/setup.sh --no-hub   # skip the Skore Hub sign-in step
#
# What it does:
#   1. Creates .venv/ at the repo root (Python 3.12+)
#   2. Installs requirements.txt and the local `parkinson` package
#   3. Installs the lab skills for Bob into .bob/skills/
#   4. Signs you in to Skore Hub and writes .skore (gitignored)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

NO_HUB=0
[[ "${1:-}" == "--no-hub" ]] && NO_HUB=1

step() { printf '\n==> %s\n' "$1"; }

# --- 1. Python + virtual environment -----------------------------------------
step "Finding Python 3.12+"
PY=""
for cand in python3.14 python3.13 python3.12 python3 python; do
    if command -v "$cand" >/dev/null 2>&1 &&
        "$cand" -c 'import sys; sys.exit(sys.version_info < (3, 12))' 2>/dev/null; then
        PY="$cand"
        break
    fi
done
if [[ -z "$PY" ]]; then
    echo "Python 3.12 or newer not found. Install it from https://www.python.org/downloads/" >&2
    exit 1
fi
echo "Using $("$PY" --version) ($(command -v "$PY"))"

if [[ ! -x .venv/bin/python ]]; then
    step "Creating .venv/"
    "$PY" -m venv .venv || {
        echo "venv creation failed. On Debian/Ubuntu: sudo apt install python3-venv" >&2
        exit 1
    }
else
    step ".venv/ already exists - reusing it"
fi
VPY="$REPO_ROOT/.venv/bin/python"

# --- 2. Packages ---------------------------------------------------------------
step "Installing requirements"
"$VPY" -m pip install --upgrade pip
"$VPY" -m pip install -r requirements.txt
"$VPY" -m pip install -e .

# --- 3. Bob skills --------------------------------------------------------------
step "Installing lab skills for Bob (.bob/skills/)"
"$REPO_ROOT/.venv/bin/skore" skills install all --repo probabl-ai/skills-hackathon --agent bob

# --- 4. Skore Hub ------------------------------------------------------------
mkdir -p data
if [[ "$NO_HUB" -eq 1 ]]; then
    step "Skipping Skore Hub sign-in (--no-hub)"
elif [[ -f .skore ]]; then
    step ".skore already exists - skipping Hub sign-in"
else
    step "Signing in to Skore Hub (a browser window will open)"
    echo "You must already be a member of the team workspace 'ibmhackathongroup1'."
    "$VPY" scripts/skore-agent
fi

# --- Check ---------------------------------------------------------------------
step "Checking the environment"
"$VPY" scripts/env_check.py

cat <<'EOF'

Setup complete.
  - Activate the environment:  source .venv/bin/activate
  - Or run without activating: ./run.sh your_script.py
  - Download the competition CSVs from the Kaggle Data tab into data/
EOF
