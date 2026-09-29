# %% [markdown]
# # 08 — Per-patient smoothing of the predictions
#
# The true target is a smooth increasing curve per patient (a quadratic in age
# fits it to ~0.2 points), but `07` predicts each visit separately, so a
# patient's predictions wiggle around that curve (~0.9 points). `PatientSmoother`
# wraps `07` and replaces each patient's predictions by a quadratic in age
# fitted through them. Compared on 07's out-of-fold predictions:
# linear 3.81, quadratic 3.68, cubic 3.70, isotonic 3.73 (raw 3.75).
#
# Hub keys: `08_smoothing` (EstimatorReport on held-out patients — the URL for
# Kaggle) and `08_smoothing_cv` (5-fold grouped CV).
#
# Run from the repo root: `python experiments/08_smoothing.py`
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
from parkinson.features import PatientPKFeatures
from parkinson.hub import load_skore_credentials
from parkinson.models import PatientSmoother

KEY = "08_smoothing"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--no-hub", action="store_true", help="Skip the Hub push.")
args, _ = parser.parse_known_args()

# %%
visits, X_test = load_visits()
X = visits.drop(columns=["Index", TARGET])
y = visits[TARGET]

model = PatientSmoother(
    make_pipeline(
        PatientPKFeatures(),
        HistGradientBoostingRegressor(
            random_state=0, max_iter=500, learning_rate=0.05, min_samples_leaf=100
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
