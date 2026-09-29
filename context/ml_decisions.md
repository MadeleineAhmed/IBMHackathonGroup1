# ML Decisions Reference

> Last updated: 2026-09-29  
> Source: probabl-ai/hackathon GUIDED.md + CONTEXT.md  
> Purpose: Running record of modelling decisions for the levodopa true-OFF regression task.  
> Each decision links back to its JOURNAL.md checklist entry.

---

## Problem Type

**REGRESSION** — predict a continuous MDS-UPDRS true-OFF motor score (range ~0–132) per patient visit.

> ⚠️ Initial assumption was binary classification. Corrected after reading the hackathon spec (J-005).

---

## Target Variable

| Property | Value | Source |
|---|---|---|
| Column | `target` | `y_train.csv` |
| Type | Continuous float | MDS-UPDRS score 0–132 |
| Meaning | Debiased "true OFF" motor score | CONTEXT.md |
| Metric | **RMSE** (lower = better) | Kaggle competition metric |
| Benchmark floor | Mean OFF + disease-duration adjustment | CONTEXT.md benchmark |

---

## Data Splitting Strategy

| Decision | Value | Rationale |
|---|---|---|
| CV splitter | `GroupKFold(n_splits=5, groups=patient_id)` | Test holdout is by patient; rows from the same patient must stay on the same side |
| **NOT** | `random row split` or `KFold` | Would leak: same patient in train AND validation fold → inflated metric |
| `random_state` | 0 everywhere | Reproducibility |
| Test set | Kaggle `X_test.csv` — never touch during development | Patient IDs do not overlap train |

### How to precompute splits (required for skore v0.26 — no `groups=` kwarg on evaluate)
```python
from sklearn.model_selection import GroupKFold
groups = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))
report = evaluate(model, X, y, splitter=cv_splits)
```

---

## Feature Set

### Columns to ALWAYS drop (leakage / identifiers)
- `Index` — row ID
- `patient_id` — group ID, not a clinical feature
- `target` — target column (only in train)

### Feature columns (from GUIDED.md)
| Column | Notes |
|---|---|
| `sexM` | Low cardinality binary |
| `age_at_diagnosis` | Numeric |
| `age` | Numeric |
| `ledd` | Levodopa equivalent daily dose — often missing |
| `time_since_intake_on` | PK timing — critical for debiasing |
| `time_since_intake_off` | PK timing — critical for debiasing |
| `on` | Biased measured ON score — often missing |
| `off` | Biased measured OFF score — often missing; **missingness is signal** |
| `cohort` | Low-cardinality string → one-hot encode |
| `gene` | High-cardinality string → StringEncoder (via skrub) |

### Derived features (to engineer)
| Feature | Formula | Rationale |
|---|---|---|
| `time_since_diagnosis` | `age - age_at_diagnosis` | Disease duration; proxy for neurodegeneration severity |

Note: `X_test` may already have `time_since_diagnosis` directly.

---

## Missingness Handling

**Do NOT impute `off` NaN with median** — missingness means the exam was skipped (patient was ON only), which is itself a signal.

Preferred approaches:
1. **`HistGradientBoostingRegressor`** — natively handles NaN by learning the best split direction for missing values. No imputation needed.
2. **`skrub.tabular_pipeline("regressor")`** — uses `TableVectorizer` + HGBR; handles mixed types and NaN automatically.
3. If imputation is tested as an ablation: use `SimpleImputer(strategy="median")` explicitly and compare RMSE.

---

## Model Progression

### Step 1 — Dummy (floor)
```python
from sklearn.dummy import DummyRegressor
from skore import evaluate
model = DummyRegressor(strategy="mean")
report = evaluate(model, X, y)
```
Hub key: `01_dummy`

### Step 2 — Ridge (linear baseline)
```python
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
model = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0))
```
Hub key: `02_ridge`

### Step 3 — HistGradientBoostingRegressor (NaN-native, numeric features)
```python
from sklearn.ensemble import HistGradientBoostingRegressor
model = HistGradientBoostingRegressor(random_state=0)
# Do NOT impute — pass NaN directly
```
Hub key: `03_hgbr`

### Step 4 — skrub tabular_pipeline (mixed types including gene/cohort)
```python
from skrub import tabular_pipeline
model = tabular_pipeline("regressor")
X_full = visits.drop(columns=["Index", "patient_id", "target"])
```
Hub key: `04_tabular_pipeline`

### Step 5 — skrub DataOps (groups baked into the graph)
```python
import skrub
data = skrub.var("visits", visits)
groups = data["patient_id"]
X_op = data.drop("target", axis=1).skb.mark_as_X(
    cv=GroupKFold(n_splits=5),
    split_kwargs={"groups": groups},
)
y_op = data["target"].skb.mark_as_y()
pred = X_op.skb.apply(skrub.TableVectorizer()).skb.apply(
    HistGradientBoostingRegressor(random_state=0), y=y_op
)
report = evaluate(pred)  # reads cv/groups from the DataOp
```
Hub key: `05_dataops`

---

## Submission Workflow

```python
from sklearn.base import clone

# Fit on ALL training data
final = clone(model).fit(X_full, y)

# Predict test
submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_test.drop(columns=["Index", "patient_id"], errors="ignore"))
submission.to_csv("submission.csv", index=False)
```

For DataOps/SkrubLearner:
```python
learner = pred.skb.make_learner()
learner.fit({"visits": visits})
pred_test = learner.predict({"visits": X_test})
```

Then upload `submission.csv` to Kaggle with the Hub report URL in the Description field.

---

## Hub Integration

```python
from parkinson.hub import load_skore_credentials
from skore import Project, login

cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="ibm-hackathon", mode="hub", workspace=cfg["workspace"])
project.put("01_dummy", report)
# Prints: Consult your report at https://skore.probabl.ai/…
```

Note: `Project.get` uses the **id** from `project.summarize()`, not the string key passed to `put`.

---

## Skore Tracking Convention

| Key pattern | Report type | When to create |
|---|---|---|
| `eda` | (reserved — do NOT use for models) | EDA only |
| `01_dummy` | DummyRegressor baseline | First submission |
| `02_ridge` | Ridge linear model | After dummy |
| `03_hgbr` | HistGBR numeric-only | After ridge |
| `04_tabular_pipeline` | skrub full mixed | After HGBR |
| `05_dataops` | skrub DataOps grouped | After tabular_pipeline |

Every Kaggle upload requires a **new** key — reusing a key is not valid.
