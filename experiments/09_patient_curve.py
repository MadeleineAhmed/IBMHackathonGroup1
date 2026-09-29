# %% [markdown]
# # 09 — Curved per-patient trend
#
# After `08` most of the error is still a per-patient level offset. The true
# target bends slightly with age (quadratic fits a patient to ~0.2 points, a
# line to ~1.2), but the patient trend features were straight lines.
# `PatientCurveFeatures` adds a per-patient quadratic of the timing-corrected
# estimates, its curvature, the spread of the estimates around the patient's
# line (how much to trust them) and the number of readings.
# Variants on grouped CV: unweighted quadratic, >= 4 readings 3.497;
# inverse-variance weighted 3.533; + separate on/off quadratics 3.537.
#
# Hub keys: `09_patient_curve` (EstimatorReport on held-out patients — the URL for
# Kaggle) and `09_patient_curve_cv` (5-fold grouped CV).
#
# Run from the repo root: `python experiments/09_patient_curve.py`
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

KEY = "09_patient_curve"

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
