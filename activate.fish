# Source this file to activate the skore pyenv virtualenv in your current fish shell.
# Usage:  source activate.fish
#         (or the short alias:  . activate.fish)

set -gx PYENV_VERSION skore
set -gx PYTHON (pyenv which python)

# Prepend the virtualenv bin to PATH so `python`, `pytest`, `pip` all resolve correctly
set -gx PATH (pyenv prefix skore)/bin $PATH

echo "✓ skore virtualenv active — $(python --version) @ $PYTHON"
