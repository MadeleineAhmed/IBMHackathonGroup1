# %% [markdown]
# # Step 13 — skrub DataOps: patient groups baked into the graph
#
# Same model as `04_tabular_pipeline` (TableVectorizer + HistGradientBoosting),
# written as a skrub DataOps graph. The GroupKFold on `patient_id` is attached
# to the feature node with `mark_as_X(cv=..., split_kwargs={"groups": ...})`,
# so the split can never drift away from the data.
#
# Note: GUIDED.md's `evaluate(pred)` raises ValueError in skore 0.26; the
# working call is `evaluate(learner, data={"visits": visits})` (cv/groups are
# still read from the DataOp).
#
# Hub keys: `05_dataops` (EstimatorReport on held-out patients — the URL for
# Kaggle) and `05_dataops_cv` (5-fold grouped CV from the DataOp).
#
# Run from the repo root: `python experiments/05_dataops.py` (`--no-hub` to
# skip the Hub push).

# %%
from __future__ import annotations

import argparse

import skrub
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from skore import EstimatorReport, Project, evaluate, login

from parkinson.data import (
    ID_COLS,
    N_SPLITS,
    TARGET,
    grouped_holdout,
    load_visits,
    write_submission,
)
from parkinson.hub import load_skore_credentials

KEY = "05_dataops"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--no-hub", action="store_true", help="Skip the Hub push.")
args, _ = parser.parse_known_args()

visits, X_test = load_visits()

# %% The graph: named input -> features (ids/target dropped) -> vectorize -> HGBR
data = skrub.var("visits", visits)
groups = data["patient_id"]  # taken before the ids are dropped
X_op = data.drop(columns=[TARGET] + ID_COLS, errors="ignore").skb.mark_as_X(
    cv=GroupKFold(n_splits=N_SPLITS),
    split_kwargs={"groups": groups},
)
y_op = data[TARGET].skb.mark_as_y()
pred = X_op.skb.apply(skrub.TableVectorizer()).skb.apply(
    HistGradientBoostingRegressor(random_state=0), y=y_op
)
learner = pred.skb.make_learner()

# %% 5-fold patient-grouped CV, read from the DataOp
cv_report = evaluate(learner, data={"visits": visits})
print(cv_report.metrics.rmse())

# %% EstimatorReport on one patient-grouped holdout (URL for Kaggle)
train_idx, test_idx = grouped_holdout(visits)
report = EstimatorReport(
    clone(learner),
    train_data={"visits": visits.iloc[train_idx]},
    test_data={"visits": visits.iloc[test_idx]},
)
print(f"Holdout RMSE: {report.metrics.rmse():.3f}")

# %% Fit on all training visits; X_test has no target, which drop(errors="ignore") allows
final = clone(learner).fit({"visits": visits})
path = write_submission(X_test, final.predict({"visits": X_test}), KEY)
print(f"Wrote {path}")

# %% Push both reports to the Hub
if not args.no_hub:
    cfg = load_skore_credentials()
    login(mode="hub")
    project = Project(name="ibm-hackathon", mode="hub", workspace=cfg["workspace"])
    project.put(f"{KEY}_cv", cv_report)
    project.put(KEY, report)
