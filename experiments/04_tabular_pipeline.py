# %% [markdown]
# # Step 12 — skrub `tabular_pipeline`: mixed types without hand-encoding
#
# Same patient-grouped CV as `03_hgbr`, but the model now also sees the text
# columns `cohort` and `gene`. `tabular_pipeline("regressor")` =
# `TableVectorizer` (picks an encoder per column) + HistGradientBoosting.
# Identifiers (`Index`, `patient_id`) are dropped before fitting.
#
# Hub keys: `04_tabular_pipeline` (EstimatorReport on held-out patients — the
# URL for Kaggle) and `04_tabular_pipeline_cv` (5-fold grouped CV).
#
# Run from the repo root: `python experiments/04_tabular_pipeline.py`
# (`--no-hub` to skip the Hub push).

# %%
from __future__ import annotations

import argparse

from sklearn.base import clone
from skore import EstimatorReport, Project, evaluate, login
from skrub import tabular_pipeline

from parkinson.data import (
    ID_COLS,
    TARGET,
    grouped_cv_splits,
    grouped_holdout,
    load_visits,
    write_submission,
)
from parkinson.hub import load_skore_credentials

KEY = "04_tabular_pipeline"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--no-hub", action="store_true", help="Skip the Hub push.")
args, _ = parser.parse_known_args()

# %% All columns except identifiers and target
visits, X_test = load_visits()
X = visits.drop(columns=ID_COLS + [TARGET])
y = visits[TARGET]
print("Features:", list(X.columns))

model = tabular_pipeline("regressor")

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
