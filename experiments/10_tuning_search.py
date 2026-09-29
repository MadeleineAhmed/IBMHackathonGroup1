# %% [markdown]
# # 10 — Settings search behind `10_tuning.py`
#
# Randomized search over HistGradientBoosting settings, 30 combinations,
# scored on the same 5 patient-grouped folds as every experiment (RMSE).
# The feature step (which learns the timing corrections per fold) is cached
# with `Pipeline(memory=...)` so each fold's features are computed once.
# Takes ~16 min. Results feed the table in `experiments/10_tuning.md`.
#
# Run from the repo root: `python experiments/10_tuning_search.py`

# %%
from __future__ import annotations

import pandas as pd
from joblib import Memory
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from parkinson import PROJECT_ROOT
from parkinson.data import TARGET, grouped_cv_splits, load_visits
from parkinson.features import PatientCurveFeatures

visits, _ = load_visits()
X = visits.drop(columns=["Index", TARGET])
y = visits[TARGET]

pipe = Pipeline(
    [("feat", PatientCurveFeatures(min_obs=4)), ("hgb", HistGradientBoostingRegressor(random_state=0))],
    memory=Memory(PROJECT_ROOT / "scratch" / "cache", verbose=0),
)
space = {
    "hgb__learning_rate": [0.02, 0.03, 0.05, 0.1],
    "hgb__max_iter": [300, 500, 800, 1200],
    "hgb__min_samples_leaf": [20, 50, 100, 200, 400],
    "hgb__max_leaf_nodes": [15, 31, 63],
    "hgb__l2_regularization": [0, 0.1, 1, 10],
    "hgb__max_features": [0.5, 0.8, 1.0],
}

# %%
search = RandomizedSearchCV(
    pipe,
    space,
    n_iter=30,
    cv=grouped_cv_splits(visits),
    scoring="neg_root_mean_squared_error",
    random_state=0,
    refit=False,
    return_train_score=True,
).fit(X, y)

results = pd.DataFrame(search.cv_results_)
results["rmse"] = -results["mean_test_score"]
results["train_rmse"] = -results["mean_train_score"]
cols = [c for c in results.columns if c.startswith("param_")] + ["rmse", "train_rmse"]
print(results.sort_values("rmse")[cols].head(10).to_string(index=False))
