# %% [markdown]
# # 07 — Pharmacokinetic correction of the readings
#
# Error analysis of `06`: 62% of the error is a per-patient offset, and
# patients with 0–1 OFF readings are much worse (RMSE 5.7 vs 3.7), because
# their curve rests on ON readings that are not corrected for dose timing.
#
# `PatientPKFeatures` learns, **on the training fold only**:
# - the ON ratio curve `on / target` by hours since intake (0.75 → 0.45),
# - the OFF bias `off - target` by hours since intake,
# then turns every reading into a target estimate, combines them per visit
# (inverse-variance weights) and fits a weighted line per patient over age.
# `min_samples_leaf=100` answers skore's SKD001 overfitting check on `06`.
#
# Hub keys: `07_pk_correction` (EstimatorReport on held-out patients — the URL
# for Kaggle) and `07_pk_correction_cv` (5-fold grouped CV).
#
# Run from the repo root: `python experiments/07_pk_correction.py`
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

KEY = "07_pk_correction"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--no-hub", action="store_true", help="Skip the Hub push.")
args, _ = parser.parse_known_args()

# %%
visits, X_test = load_visits()
X = visits.drop(columns=["Index", TARGET])
y = visits[TARGET]

model = make_pipeline(
    PatientPKFeatures(),  # corrections fitted inside each training fold
    HistGradientBoostingRegressor(
        random_state=0, max_iter=500, learning_rate=0.05, min_samples_leaf=100
    ),
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
