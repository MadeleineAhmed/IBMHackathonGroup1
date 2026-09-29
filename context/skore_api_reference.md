# Skore API Reference (v0.26.0)

> Last updated: 2026-09-29
> Source: PyPI skore 0.26.0 + skore-cli 0.4.1, github.com/probabl-ai/hackathon, live introspection
> Purpose: Quick reference for the skore classes and Hub integration used in this project.

---

## Installation

```bash
pip install -U skore          # base
pip install -U skore[hub]     # + Skore Hub sync
pip install -U skore[mlflow]  # + MLflow logging
```

Active environment: `pyenv virtualenv skore` (Python 3.14.7)

---

## Public API — top-level imports

```python
from skore import (
    CrossValidationReport,
    EstimatorReport,
    ComparisonReport,
    TrainTestSplit,
    evaluate,
    compare,
    configuration,
    Project,
    Summary,
)
```

---

## TrainTestSplit

**sklearn-compatible splitter** (not a convenience function — it yields index arrays).

```python
from skore import TrainTestSplit

tts = TrainTestSplit(test_size=0.2, random_state=42)

# split() is a GENERATOR that yields (train_idx, test_idx) — sklearn splitter protocol
train_idx, test_idx = next(tts.split(X, y))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
```

**Signature:**
```
TrainTestSplit(
    test_size: float | int | None = 0.2,
    train_size: float | int | None = None,
    random_state: int | RandomState | None = 0,
    shuffle: bool = True,
    stratify: ArrayLike | None = None,
)
```

> ⚠️ Do NOT unpack `tts.split(X, y)` as 4 values — it returns a generator of `(train_idx, test_idx)` tuples.

---

## EstimatorReport

Single model evaluation on a pre-split train/test set.

```python
from skore import EstimatorReport

report = EstimatorReport(
    estimator,           # any sklearn-compatible estimator (unfitted)
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)

# Inspect available metrics
report.help()

# Get metrics summary as a DataFrame
df = report.metrics.summarize().frame()

# Individual metrics
report.metrics.roc()             # ROC curve display
report.metrics.precision_recall()
report.metrics.confusion_matrix()
```

---

## CrossValidationReport

Cross-validated evaluation.

```python
from skore import CrossValidationReport

report = CrossValidationReport(
    estimator,           # unfitted estimator
    X,
    y,
    splitter=5,          # int (n_folds) OR sklearn CV splitter object
    n_jobs=None,         # parallel jobs
)

# ⚠️ The parameter is `splitter=`, NOT `cv=`

df = report.metrics.summarize().frame()
```

**Signature:**
```
CrossValidationReport(
    estimator: EstimatorLike,
    X: ArrayLike | None = None,
    y: ArrayLike | None = None,
    data: dict | None = None,
    pos_label: PositiveLabel | None = None,
    splitter: int | SKLearnCrossValidator | Generator | None = None,
    n_jobs: int | None = None,
)
```

---

## ComparisonReport

Compare multiple `EstimatorReport` objects side by side.

```python
from skore import ComparisonReport

r1 = EstimatorReport(Ridge(), X_train=..., y_train=..., X_test=..., y_test=...)
r2 = EstimatorReport(HistGradientBoostingRegressor(), X_train=..., y_train=..., X_test=..., y_test=...)

comp = ComparisonReport([r1, r2])      # or a dict {"ridge": r1, "hgbr": r2}

df = comp.metrics.summarize().frame()
```

A `ComparisonReport` **cannot** be pushed with `project.put` — push `comp.reports_[name]` instead.

---

## evaluate / compare (high-level dispatch)

`evaluate` returns a different report type depending on what you pass:

| Call | Returns |
|---|---|
| `evaluate(est, X, y)` (default `splitter=0.2`) | `EstimatorReport` — single **random row** holdout |
| `evaluate(est, X, y, splitter=cv_splits)` (list of index pairs, int, or CV object) | `CrossValidationReport` |
| `evaluate({"a": est1, "b": est2}, X, y, ...)` (dict or list of estimators) | `ComparisonReport` |
| `evaluate(fitted_est, X_test, y_test, splitter="prefit")` | `EstimatorReport` on an already-fitted model |

```python
from skore import evaluate, compare

report = evaluate(estimator, X, y, splitter=cv_splits)

# compare: takes REPORTS you already built (not estimators + X, y)
comp = compare([report_a, report_b])          # or {"a": report_a, "b": report_b}

# get one model's report back out of a ComparisonReport
ridge_report = comp.reports_["ridge"]         # reports_ is a dict keyed by name
```

Regression metrics on a report: `report.metrics.rmse()`, `.mae()`, `.r2()`, `.mape()`, `.summarize()`.

---

## Project (persistent experiment tracking)

```python
from skore import Project

project = Project(name="ibm-hackathon", mode="hub", workspace=cfg["workspace"])  # see Hub section

project.put("02_ridge", report)   # ONLY EstimatorReport or CrossValidationReport
project.summarize()               # table of stored reports (with their ids)
project.get(report_id)            # by the id from summarize(), NOT by the put key
project.delete(...)               # remove a report
```

> ⚠️ `project.put(key, comparison_report)` raises `TypeError`. To push a model from a
> comparison, put its sub-report: `project.put("02_ridge", comp.reports_["ridge"])`.
> There is no `list_item_keys()` in v0.26.

---

## configuration

Global configuration object for skore behaviour.

```python
from skore import configuration

# Inspect available settings
print(configuration)
```

---

## Common Patterns for This Project

### Patient-grouped CV (required — the Kaggle holdout is by patient)
skore calls `splitter.split(X, y)` without `groups=`, so precompute the index pairs:
```python
from sklearn.model_selection import GroupKFold
from skore import evaluate

groups = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))
report = evaluate(model, X, y, splitter=cv_splits)   # -> CrossValidationReport
```

### Metrics access pattern
```python
report.metrics.rmse()                  # competition metric
df = report.metrics.summarize().frame()
# Returns a pandas DataFrame with metric names as index/columns
```

---

## Hub Integration (hackathon-specific)

Workspace: [`ibmhackathongroup1`](https://skore.probabl.ai/ibmhackathongroup1) ·
Project: [`ibm-hackathon`](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon)

### Prerequisites
All handled by the team setup script (`setup/windows/setup.bat` or `setup/unix/setup.sh`):
1. `skore-cli` installed (pinned in `requirements.txt`)
2. `.skore` written at this repo's root by `python scripts/skore-agent` (gitignored — holds the API key)
3. Local `parkinson` package installed editable (`pip install -e .`) — `src/parkinson/hub.py`
   is the `organize-ml-workspace` skill's `templates/src_hub.py`

### Hub setup (run once per machine)
```bash
# From this repo's root (you must already be a member of the workspace):
python scripts/skore-agent
# Opens browser → sign in at https://skore.probabl.ai → writes .skore
```

### Hub push pattern (required for every Kaggle submission)
```python
from parkinson.hub import load_skore_credentials  # src/parkinson/hub.py
from skore import Project, login

cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="ibm-hackathon", mode="hub", workspace=cfg["workspace"])
project.put("01_dummy", report)
# Console prints: Consult your report at https://skore.probabl.ai/…
# Copy that URL into the Kaggle Submission Description
```

### Key rules
- Use a **new key** for each Kaggle submission (`01_dummy`, `02_ridge`, etc.)
- `eda` key is **reserved** for EDA — do NOT use it for model reports
- `Project.get()` uses the **id** from `project.summarize()`, not the string key
- Project name is **`ibm-hackathon`**. The skill doc
  `.bob/skills/organize-ml-workspace/references/hub_credentials.md` spells it
  `ibm-hackaton` (typo) — copying that would create a second, separate project.
- Only `EstimatorReport` / `CrossValidationReport` can be `put` (see Project section).
