# %% [markdown]
# # 06 — Patient-level features (beyond the guide)
#
# The debiased target is a smooth, increasing curve per patient; each visit's
# `on` / `off` is a noisy reading of it. Steps 10–13 predict every visit
# alone and plateau near 7.4. Here `PatientFeatures` adds, for each visit,
# summaries of the same patient's other visits (mean / median / min / max /
# count of `off`, `on`, `ledd`; per-patient linear trend and slope of `off`
# and `on` over age; visit position). Input columns only — never the target.
#
# Hub keys: `06_patient_features` (EstimatorReport on held-out patients — the
# URL for Kaggle) and `06_patient_features_cv` (5-fold grouped CV).
#
# Run from the repo root: `python experiments/06_patient_features.py`
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
from parkinson.features import PatientFeatures
from parkinson.hub import load_skore_credentials

KEY = "06_patient_features"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--no-hub", action="store_true", help="Skip the Hub push.")
args, _ = parser.parse_known_args()

# %% Raw visit columns in (patient_id is needed to group; the transformer drops it)
visits, X_test = load_visits()
X = visits.drop(columns=["Index", TARGET])
y = visits[TARGET]

model = make_pipeline(
    PatientFeatures(),
    HistGradientBoostingRegressor(random_state=0, max_iter=500, learning_rate=0.05),
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
