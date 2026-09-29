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

r1 = EstimatorReport(LogisticRegression(), X_train=..., y_train=..., X_test=..., y_test=...)
r2 = EstimatorReport(RandomForestClassifier(), X_train=..., y_train=..., X_test=..., y_test=...)

comp = ComparisonReport([r1, r2])

df = comp.metrics.summarize().frame()
```

---

## evaluate / compare (high-level dispatch)

```python
from skore import evaluate, compare

# evaluate: auto-detects task type and returns the right report
report = evaluate(estimator, X, y)

# compare: multiple estimators at once
comp = compare([est1, est2], X, y)
```

---

## Project (persistent experiment tracking)

```python
from skore import Project

project = Project("my_experiment")   # creates/opens a skore project file

project.put("cv_report", cv_report)  # store any report or object
project.get("cv_report")             # retrieve it
project.list_item_keys()             # see all stored keys
```

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

### Stratified CV (important for imbalanced OFF/ON classes)
```python
from sklearn.model_selection import StratifiedKFold
from skore import CrossValidationReport

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
report = CrossValidationReport(estimator, X, y, splitter=cv)
```

### Setting positive label for OFF-state recall focus
```python
report = CrossValidationReport(
    estimator, X, y,
    splitter=cv,
    pos_label="OFF",   # ensures recall/precision computed for OFF class
)
```

### Metrics access pattern
```python
df = report.metrics.summarize().frame()
# Returns a pandas DataFrame with metric names as index/columns
```

---

## Hub Integration (hackathon-specific)

### Prerequisites
1. `skore-cli` is installed: `pip install --upgrade skore-cli`
2. `.skore` file created by running `python scripts/skore-agent` from the hackathon repo root
3. `parkinson/` package present (comes from the hackathon repo clone — contains `hub.py`)

### Hub setup (run once per machine)
```bash
# From the cloned hackathon repo root:
python scripts/skore-agent
# Opens browser → sign in at https://skore.probabl.ai → writes .skore
```

### Hub push pattern (required for every Kaggle submission)
```python
from parkinson.hub import load_skore_credentials  # from hackathon repo
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

### skore-cli internals (if needed without parkinson package)
```python
from skore_cli.agent._skore_file import SkoreConfig  # NOT skore_cli._skore_file
from skore_cli._hub_auth import ensure_login
from skore_cli.agent._commands import _resolve_membership
```
