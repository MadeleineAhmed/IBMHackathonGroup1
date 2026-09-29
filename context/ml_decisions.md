# ML Decisions Reference

> Last updated: 2026-09-29  
> Source: probabl-ai/hackathon GUIDED.md + CONTEXT.md  
> Purpose: Running record of modelling decisions for the levodopa true-OFF regression task.  
> Experiments are tracked in `journal/JOURNAL.md`; setup history in `journal/setup_log.md`.

---

## Problem Type

**REGRESSION** — predict a continuous MDS-UPDRS true-OFF motor score (range ~0–132) per patient visit.

> ⚠️ Initial assumption was binary classification. Corrected after reading the hackathon spec (setup log J-005).

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
report = evaluate(model, X, y)   # default splitter=0.2: random ROW holdout (as in GUIDED.md)
```
Hub key: `01_dummy`

> Steps 1–2 follow the guide and use the default random row holdout, which leaks
> patients across the split. Their RMSE is optimistic and **not comparable** with
> steps 3+. From step 3 on, always pass `splitter=cv_splits` (GUIDED.md step 10).

### Step 2 — Ridge (linear baseline)
```python
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
model = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0))
report = evaluate(model, X, y)

# comparing dummy vs ridge returns a ComparisonReport, which put() rejects:
comp = evaluate({"dummy": dummy, "ridge": model}, X, y)
report = comp.reports_["ridge"]  # push this one
```
Hub key: `02_ridge`

### Step 3 — HistGradientBoostingRegressor (NaN-native, numeric features)
```python
from sklearn.ensemble import HistGradientBoostingRegressor
model = HistGradientBoostingRegressor(random_state=0)
# Do NOT impute — pass NaN directly. X = numeric feature_cols only
# (gene / cohort are strings: add them later, or .astype("category"))
report = evaluate(model, X, y, splitter=cv_splits)
```
Hub key: `03_hgbr`

### Step 4 — skrub tabular_pipeline (mixed types including gene/cohort)
```python
from skrub import tabular_pipeline
model = tabular_pipeline("regressor")
X_full = visits.drop(columns=["Index", "patient_id", "target"])
report = evaluate(model, X_full, y, splitter=cv_splits)
```
Hub key: `04_tabular_pipeline`

### Step 5 — skrub DataOps (groups baked into the graph)
```python
import skrub
data = skrub.var("visits", visits)
groups = data["patient_id"]
# drop ids too: Index / patient_id must not be features (errors="ignore": X_test has no target)
X_op = data.drop(columns=["target", "patient_id", "Index"], errors="ignore").skb.mark_as_X(
    cv=GroupKFold(n_splits=5),
    split_kwargs={"groups": groups},
)
y_op = data["target"].skb.mark_as_y()
pred = X_op.skb.apply(skrub.TableVectorizer()).skb.apply(
    HistGradientBoostingRegressor(random_state=0), y=y_op
)
learner = pred.skb.make_learner()
# NOTE: GUIDED.md's `evaluate(pred)` raises ValueError in skore 0.26 —
# pass the learner plus its data bindings; cv/groups still come from the DataOp.
report = evaluate(learner, data={"visits": visits})  # -> CrossValidationReport
```
Hub key: `05_dataops`

---

## Submission Workflow

```python
from sklearn.base import clone

# X = the exact feature matrix the model was evaluated on
# (X[feature_cols] for dummy/ridge/hgbr, X_full for tabular_pipeline)
final = clone(model).fit(X, y)

# Predict test with the same columns
submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_test[X.columns])
submission.to_csv("submission.csv", index=False)
```

For DataOps/SkrubLearner:
```python
learner = pred.skb.make_learner()
learner.fit({"visits": visits})
submission = X_test[["Index"]].copy()
submission["target"] = learner.predict({"visits": X_test})
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

Notes:
- Project name is `ibm-hackathon` (the skill doc's `ibm-hackaton` is a typo).
- `put` only accepts `EstimatorReport` / `CrossValidationReport`; for a `ComparisonReport`
  push `comp.reports_["<name>"]`.
- `Project.get` uses the **id** from `project.summarize()`, not the string key passed to `put`.

---

## Skore Tracking Convention

| Key pattern | Report type | When to create |
|---|---|---|
| `eda` | (reserved — do NOT use for models) | EDA only |
| `01_dummy` | DummyRegressor baseline | First submission |
| `02_ridge` | Ridge linear model | After dummy |
| `03_hgbr` | HistGBR numeric-only (grouped CV) | After ridge |
| `04_tabular_pipeline` | skrub full mixed | After HGBR |
| `05_dataops` | skrub DataOps grouped | After tabular_pipeline |

Every Kaggle upload requires a **new** key — reusing a key is not valid.

---

## Beyond the guide (where we can beat other teams)

The 5 steps above are the same for every team. The edge comes from encoding how the
target was built. Each idea is a Backlog row in `journal/JOURNAL.md`; judge every one
by **grouped-CV RMSE**, not the public leaderboard.

| # | Idea | Why it should help |
|---|---|---|
| B6 | **Patient-level aggregates from X only** — per-patient mean/median/min/max of `on`/`off`, slope of observed scores vs `time_since_diagnosis`, number of visits, visit position, share of visits with `off` missing | "True OFF" is a smooth per-patient progression; one visit is noisy, all of a patient's visits are not. Test patients also have several visits in `X_test`, so this is legal (no target used). Compute inside each CV fold. |
| B7 | **Pharmacokinetic features** — residual-drug proxy `exp(-time_since_intake_off / t_half)` (t½ ≈ 1–1.5 h), `on`/`off` gap and ratio, `ledd`-scaled versions | Observed OFF is biased by drug still in the blood; ON typically improves 50–100% over OFF. Gives trees the right shape instead of making them find it. |
| B8 | **Monotonicity** — `HistGradientBoostingRegressor(monotonic_cst=...)` increasing in disease duration; per-patient isotonic/smooth post-processing of predictions | Neurodegeneration only worsens; a lone jumpy visit prediction is almost surely error. |
| B9 | **Mixed-effects model** — patient random intercept + slope on disease duration | Classic longitudinal model; strong, explainable, good stacking input. |
| B10 | **Missingness patterns** — indicators for which of `on`/`off`/`ledd`/timing are missing, per visit and per patient | The pattern encodes cohort protocol and patient state, beyond "off is missing". |
| B11 | **Error analysis** — residuals by cohort / gene / missingness / visit position from skore reports (`iterate-from-skore`, `audit-ml-pipeline` skills) | Target features where the model is wrong instead of blind tuning. |
| B12 | **Ensembling** — average/stack the best grouped-CV models | Small, reliable final gain once features are good. |

First EDA question: how many visits per patient, and how smooth is `target` across a
patient's visits? If it looks like a clean per-patient curve, B6 and B8 are the big wins.

