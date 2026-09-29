# %% [markdown]
# # Steps 10–11 — Patient-grouped CV + HistGradientBoosting
#
# - Evaluation switches to **GroupKFold by `patient_id`** (5 folds): a
#   patient's visits are never on both sides, like the Kaggle test set.
# - `HistGradientBoostingRegressor` on the 8 numeric features, with **no
#   imputation**: missing values are routed at each split, so "OFF exam
#   skipped" becomes a learnable signal.
#
# Hub keys: `03_hgbr` (EstimatorReport on held-out patients — the URL for the
# Kaggle Submission Description) and `03_hgbr_cv` (5-fold grouped CV).
#
# Run from the repo root: `python experiments/03_hgbr.py` (`--no-hub` to skip
# the Hub push).

# %%
from __future__ import annotations

import argparse

from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from skore import EstimatorReport, Project, evaluate, login

from parkinson.data import (
    NUMERIC_COLS,
    TARGET,
    grouped_cv_splits,
    grouped_holdout,
    load_visits,
    write_submission,
)
from parkinson.hub import load_skore_credentials

KEY = "03_hgbr"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--no-hub", action="store_true", help="Skip the Hub push.")
args, _ = parser.parse_known_args()

# %%
visits, X_test = load_visits()
X = visits[NUMERIC_COLS]
y = visits[TARGET]

model = HistGradientBoostingRegressor(random_state=0)

# %% 5-fold patient-grouped CV (the score we compare models on)
cv_report = evaluate(model, X, y, splitter=grouped_cv_splits(visits))
print(cv_report.metrics.rmse())

# %% EstimatorReport on one patient-grouped holdout (URL for Kaggle)
train_idx, test_idx = grouped_holdout(visits)
report = EstimatorReport(
    clone(model),
    X_train=X.iloc[train_idx],
    y_train=y.iloc[train_idx],
    X_test=X.iloc[test_idx],
    y_test=y.iloc[test_idx],
)
print(f"Holdout RMSE: {report.metrics.rmse():.3f}")

# %% Fit on all training visits and write the Kaggle submission
final = clone(model).fit(X, y)
path = write_submission(X_test, final.predict(X_test[NUMERIC_COLS]), KEY)
print(f"Wrote {path}")

# %% Push both reports to the Hub
if not args.no_hub:
    cfg = load_skore_credentials()
    login(mode="hub")
    project = Project(name="ibm-hackathon", mode="hub", workspace=cfg["workspace"])
    project.put(f"{KEY}_cv", cv_report)
    project.put(KEY, report)
