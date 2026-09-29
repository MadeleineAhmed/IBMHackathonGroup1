# Activate the project environment in the current fish shell.
# Usage (from the repo root):  source setup/unix/activate.fish
#
# Uses .venv/ (created by setup/unix/setup.sh) when present, otherwise the
# 'skore' pyenv virtualenv.

set -l repo_root (realpath (dirname (status filename))/../..)

if test -f $repo_root/.venv/bin/activate.fish
    source $repo_root/.venv/bin/activate.fish
else
    set -gx PYENV_VERSION skore
    set -gx PATH (pyenv prefix skore)/bin $PATH
end

echo "✓ environment active — "(python --version)" @ "(command -v python)
