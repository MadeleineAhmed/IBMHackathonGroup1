# %% [markdown]
# # 10 — Tuning the model settings
#
# Same features and smoothing as `09`; only the HistGradientBoosting settings
# change. Chosen by a randomized search (30 combinations) scored on the same
# 5 patient-grouped folds: learning rate, number of trees, leaves per tree,
# minimum visits per leaf, share of columns per tree, L2 penalty.
# Top 3 with smoothing: 3.467 / 3.468 / 3.469 vs 3.497 for 09's settings.
# Column selection (dropping sexM / gene / cohort flagged by skore SKD012)
# made no difference (3.498), so all columns are kept.
#
# Hub keys: `10_tuning` (EstimatorReport on held-out patients — the URL for
# Kaggle) and `10_tuning_cv` (5-fold grouped CV).
#
# Run from the repo root: `python experiments/10_tuning.py`
# (`--no-hub` to skip the Hub push).

# %%
from __future__ import annotations

import argparse

from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from skore import EstimatorReport, Project, evaluate, login

from parkinson.data import (
    TARGET,
    grouped_cv_splits,
    grouped_holdout,
    load_visits,
    write_submission,
)
from parkinson.features import PatientCurveFeatures
from parkinson.hub import load_skore_credentials
from parkinson.models import PatientSmoother

KEY = "10_tuning"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--no-hub", action="store_true", help="Skip the Hub push.")
args, _ = parser.parse_known_args()

# %%
visits, X_test = load_visits()
X = visits.drop(columns=["Index", TARGET])
y = visits[TARGET]

model = PatientSmoother(
    make_pipeline(
        PatientCurveFeatures(min_obs=4),
        HistGradientBoostingRegressor(
            random_state=0,
            learning_rate=0.02,  # learn slowly...
            max_iter=1200,  # ...with more trees
            max_leaf_nodes=63,
            min_samples_leaf=200,  # each rule must cover >= 200 visits
            max_features=0.8,  # each split sees 80% of the columns
            l2_regularization=0.0,
        ),
    ),
    degree=2,
)

# %% 5-fold patient-grouped CV
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
path = write_submission(X_test, final.predict(X_test[X.columns]), KEY)
print(f"Wrote {path}")

# %% Push both reports to the Hub
if not args.no_hub:
    cfg = load_skore_credentials()
    login(mode="hub")
    project = Project(name="ibm-hackathon", mode="hub", workspace=cfg["workspace"])
    project.put(f"{KEY}_cv", cv_report)
    project.put(KEY, report)
