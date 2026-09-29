# %% [markdown]
# # 11 — Ensemble
#
# Averages 4 models built on the same `PatientCurveFeatures`, then applies the
# per-patient smoothing once (a per-patient polynomial fit is linear, so
# smoothing the average = averaging the smoothed predictions):
# - the top-3 HistGradientBoosting settings from `10_tuning_search.py`
#   (3.467 / 3.468 / 3.469 alone), 30% each;
# - a Ridge on the same features (3.99 alone), 10%: much weaker, but its
#   errors differ (correlation 0.81 with the trees), which the average uses.
# Grouped CV of the out-of-fold blends: mean of 3 HGBR 3.437, + 10% Ridge
# 3.430, + 20% 3.438; non-negative least squares also picked ~10% Ridge.
#
# Hub keys: `11_ensemble` (EstimatorReport on held-out patients — the URL for
# Kaggle) and `11_ensemble_cv` (5-fold grouped CV).
#
# Run from the repo root: `python experiments/11_ensemble.py`
# (`--no-hub` to skip the Hub push).

# %%
from __future__ import annotations

import argparse

from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
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

KEY = "11_ensemble"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--no-hub", action="store_true", help="Skip the Hub push.")
args, _ = parser.parse_known_args()

# %%
visits, X_test = load_visits()
X = visits.drop(columns=["Index", TARGET])
y = visits[TARGET]



def hgb(**settings):
    return make_pipeline(
        PatientCurveFeatures(min_obs=4),
        HistGradientBoostingRegressor(random_state=0, **settings),
    )


model = PatientSmoother(
    VotingRegressor(
        [
            ("hgb1", hgb(learning_rate=0.02, max_iter=1200, max_leaf_nodes=63,
                         min_samples_leaf=200, max_features=0.8)),
            ("hgb2", hgb(learning_rate=0.03, max_iter=1200, max_leaf_nodes=31,
                         min_samples_leaf=50, l2_regularization=1.0)),
            ("hgb3", hgb(learning_rate=0.05, max_iter=500, max_leaf_nodes=63,
                         min_samples_leaf=20, max_features=0.5)),
            ("ridge", make_pipeline(
                PatientCurveFeatures(min_obs=4),
                SimpleImputer(strategy="median", add_indicator=True),
                StandardScaler(),
                Ridge(alpha=1.0),
            )),
        ],
        weights=[0.3, 0.3, 0.3, 0.1],
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
