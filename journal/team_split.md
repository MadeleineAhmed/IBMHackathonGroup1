# Team Work Split — IBM × Probabl Hackathon

> ⚠️ **PRELIMINARY** — assignments not yet confirmed with the full team. Update owners and status once agreed.

> Update the **Status** column and **Owner** as work progresses.
> All code goes in `experiments/NN_name.py`. Every submission needs a Hub URL in the Kaggle Description.  
> Shared data objects (`visits`, `X`, `y`, `feature_cols`) come from Step 7 — no one else can run `evaluate` until that is done.

---

## Critical path

```
Person 1: Step 7 (dummy baseline) ──────────────────────────────► unblocks everyone
                                    │
                                    ├──► Person 2: Steps 8–9   (Ridge + Kaggle submit)
                                    ├──► Person 3: Steps 10–11 (GroupKFold + HGBR)
                                    └──► Person 4: Steps 12–13 (tabular_pipeline + DataOps)
```

Persons 2, 3, 4 start the moment Person 1 confirms:
- Hub push printed a URL ✓
- `submission.csv` uploaded to Kaggle ✓
- Shared the data load snippet (see Step 7 below)

---

## Overview table

| Person | Steps | Script | Hub key(s) | Status | Owner |
|--------|-------|--------|------------|--------|-------|
| 1 | 7 — Dummy baseline | `experiments/01_dummy.py` | `01_dummy` | ⏳ Pending | |
| 2 | 8–9 — Ridge + submission | `experiments/02_ridge.py` | `02_ridge` | ⏳ Blocked on Step 7 | |
| 3 | 10–11 — GroupKFold + HGBR | `experiments/03_hgbr.py` | `03_hgbr` | ⏳ Blocked on Step 7 | |
| 4 | 12–13 — tabular_pipeline + DataOps | `experiments/04_tabular_pipeline.py`, `experiments/05_dataops.py` | `04_tabular_pipeline`, `05_dataops` | ⏳ Blocked on Step 7 | |

---

## Person 1 — Step 7: Dummy baseline

**Goal:** prove the full pipeline works end-to-end (data load → evaluate → Hub → Kaggle).  
**Script:** `experiments/01_dummy.py`  
**Hub key:** `01_dummy`

### What to do

1. Load the data and merge train tables:
   ```python
   import pandas as pd

   X_train = pd.read_csv("data/X_train.csv")
   y_train = pd.read_csv("data/y_train.csv")
   X_test  = pd.read_csv("data/X_test.csv")

   visits = X_train.merge(y_train, on="Index")

   feature_cols = [
       "sexM", "age_at_diagnosis", "age", "ledd",
       "time_since_intake_on", "time_since_intake_off", "on", "off",
   ]
   X = visits[feature_cols]
   y = visits["target"]
   ```

2. Run the dummy model (predicts the training mean for every row — ignores all features):
   ```python
   from sklearn.dummy import DummyRegressor
   from skore import evaluate

   dummy = DummyRegressor(strategy="mean")
   report = evaluate(dummy, X, y)   # default splitter=0.2 random row holdout — fine for Step 7
   report.metrics.rmse()
   ```

3. Push to Hub:
   ```python
   from parkinson.hub import load_skore_credentials
   from skore import Project, login

   cfg = load_skore_credentials()
   login(mode="hub")
   project = Project(name="ibm-hackathon", mode="hub", workspace=cfg["workspace"])
   project.put("01_dummy", report)
   # Console prints: Consult your report at https://skore.probabl.ai/…  ← copy this URL
   ```

4. Generate and upload `submission.csv`:
   ```python
   from sklearn.base import clone

   final = clone(dummy).fit(X, y)
   submission = X_test[["Index"]].copy()
   submission["target"] = final.predict(X_test[feature_cols])
   submission.to_csv("submission.csv", index=False)
   ```
   Upload on Kaggle. Paste the Hub URL into the Submission Description.

5. **Share with the team:** the `feature_cols` list, `visits`, `X`, `y` (or just the script above).

### Why this exists
If you cannot beat predicting the mean, the data load, metric, or upload is broken — not the model. This is the floor every other model must beat.

---

## Person 2 — Steps 8–9: Ridge + first real Kaggle submission

**Goal:** check if numeric features carry any linear signal; produce first non-trivial submission.  
**Script:** `experiments/02_ridge.py`  
**Hub key:** `02_ridge`  
**Blocked until:** Person 1 confirms Hub push + shares data load

### What to do

1. Copy Person 1's data load block (`visits`, `X`, `y`, `feature_cols`).

2. Build the Ridge pipeline (Ridge needs no NaNs, so impute medians first):
   ```python
   from sklearn.impute import SimpleImputer
   from sklearn.linear_model import Ridge
   from sklearn.pipeline import make_pipeline
   from skore import evaluate

   ridge = make_pipeline(
       SimpleImputer(strategy="median"),
       Ridge(alpha=1.0),
   )
   report = evaluate(ridge, X, y)
   report.metrics.rmse()
   ```

3. Compare against dummy in one report (optional but informative):
   ```python
   comp = evaluate({"dummy": dummy, "ridge": ridge}, X, y)
   # comp is a ComparisonReport — project.put() rejects it directly
   # push only the ridge sub-report:
   report = comp.reports_["ridge"]
   ```

4. Tune `alpha` — try `0.1`, `1.0`, `10.0`, keep the one with lowest RMSE:
   ```python
   for alpha in [0.1, 1.0, 10.0]:
       m = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=alpha))
       r = evaluate(m, X, y)
       print(alpha, r.metrics.rmse())
   ```

5. Push best result and submit to Kaggle:
   ```python
   project.put("02_ridge", report)   # prints Hub URL — copy it

   final = clone(ridge).fit(X, y)
   submission = X_test[["Index"]].copy()
   submission["target"] = final.predict(X_test[feature_cols])
   submission.to_csv("submission_02_ridge.csv", index=False)
   ```

### Gotcha
`project.put(key, comp)` raises `TypeError` — `ComparisonReport` is not accepted. Always push `comp.reports_["ridge"]`.

---

## Person 3 — Steps 10–11: GroupKFold CV + HGBR

**Goal:** switch to the honest patient-grouped evaluation and introduce a tree model that handles missing values natively.  
**Script:** `experiments/03_hgbr.py`  
**Hub key:** `03_hgbr`  
**Blocked until:** Person 1 confirms Hub push + shares data load

### What to do

**Step 10 — GroupKFold (the correct evaluation from here on)**

Steps 7–9 use a random row holdout, which leaks: the same patient's visits appear in both train and validation, so the metric looks better than it really is. From Step 10 onward the split is by patient.

```python
from sklearn.model_selection import GroupKFold
from skore import evaluate

groups = visits["patient_id"]   # one group label per row
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))
# cv_splits is a list of (train_indices, val_indices) pairs — 5 folds
```

Why precompute? skore 0.26 calls `splitter.split(X, y)` without `groups=`, so passing the splitter object directly would drop the patient grouping. Precomputing the pairs locks groups in.

**Step 11 — HistGradientBoostingRegressor**

`off` being missing is not noise — it usually means the uncomfortable OFF exam was skipped (patient was ON only). Ridge had to fill those holes with a median. HGBR learns which direction to send missing values at each split, so the missingness itself becomes signal.

```python
from sklearn.ensemble import HistGradientBoostingRegressor

hgbr = HistGradientBoostingRegressor(random_state=0)
# Do NOT impute — pass NaN directly
report = evaluate(hgbr, X, y, splitter=cv_splits)
report.metrics.rmse()
project.put("03_hgbr", report)   # prints Hub URL
```

Fit on all training data and submit:
```python
final = clone(hgbr).fit(X, y)
submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_test[feature_cols])
submission.to_csv("submission_03_hgbr.csv", index=False)
```

### Why GroupKFold matters
The Kaggle test set holds out **entire patients** — no test patient appears in train. A random row holdout gives an optimistic RMSE because the same patient is on both sides. GroupKFold replicates the actual test condition.

---

## Person 4 — Steps 12–13: skrub tabular_pipeline + DataOps

**Goal:** include `gene` and `cohort` (string columns that Ridge and HGBR ignored) using skrub's automatic mixed-type encoding, then bake the grouped split into the pipeline graph.  
**Scripts:** `experiments/04_tabular_pipeline.py`, `experiments/05_dataops.py`  
**Hub keys:** `04_tabular_pipeline`, `05_dataops`  
**Blocked until:** Person 1 confirms Hub push + shares data load  
**Useful reference:** Person 3's `cv_splits` pattern (or copy it directly)

### What to do

**Step 12 — skrub `tabular_pipeline`**

Ridge and HGBR above only saw numeric columns. `gene` and `cohort` are strings that could carry real signal (different genetic cohorts have different disease trajectories). `tabular_pipeline("regressor")` wraps `TableVectorizer` + HGBR: it picks an encoder per column automatically (one-hot for low-cardinality, `StringEncoder` for high-cardinality text, passthrough for numbers).

```python
from skrub import tabular_pipeline
from skore import evaluate

# Include all columns except identifiers and target
X_full = visits.drop(columns=["Index", "patient_id", "target"])
model = tabular_pipeline("regressor")
report = evaluate(model, X_full, y, splitter=cv_splits)
report.metrics.rmse()
project.put("04_tabular_pipeline", report)
```

Submission:
```python
final = clone(model).fit(X_full, y)
submission = X_test[["Index"]].copy()
submission["target"] = final.predict(
    X_test.drop(columns=["Index", "patient_id"], errors="ignore")
)
submission.to_csv("submission_04_tabular_pipeline.csv", index=False)
```

**Step 13 — skrub DataOps**

Instead of keeping `cv_splits` as a separate variable that can go stale, DataOps bakes the GroupKFold and the `patient_id` groups directly onto the feature node of the graph. The split cannot drift away from the data.

```python
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from skore import evaluate

data   = skrub.var("visits", visits)
groups = data["patient_id"]
X_op   = data.drop(columns=["target", "patient_id", "Index"], errors="ignore").skb.mark_as_X(
    cv=GroupKFold(n_splits=5),
    split_kwargs={"groups": groups},
)
y_op = data["target"].skb.mark_as_y()
pred = X_op.skb.apply(skrub.TableVectorizer()).skb.apply(
    HistGradientBoostingRegressor(random_state=0), y=y_op
)

learner = pred.skb.make_learner()
report  = evaluate(learner, data={"visits": visits})   # cv/groups come from the DataOp
project.put("05_dataops", report)
```

Submission (DataOps uses a dict environment, not bare arrays):
```python
learner.fit({"visits": visits})
submission = X_test[["Index"]].copy()
submission["target"] = learner.predict({"visits": X_test})
submission.to_csv("submission_05_dataops.csv", index=False)
```

### Gotchas
- `evaluate(pred)` (passing the DataOp directly) raises `ValueError` in skore 0.26 — pass `learner` + `data=` instead.
- `project.put(key, comp)` still rejects `ComparisonReport` — same rule as Person 2.

---

## Shared rules (everyone)

| Rule | Detail |
|------|--------|
| **Group by patient** | From Step 10 onward always use `splitter=cv_splits` (GroupKFold on `patient_id`) |
| **No id features** | Drop `patient_id` and `Index` before fitting |
| **Keep missing values** | Do not impute `off` — missingness is signal (exception: Ridge in Step 8 must impute) |
| **Hub URL on every submission** | Paste the printed URL into the Kaggle Submission Description or the submission is invalid |
| **Submission format** | CSV with header `Index,target`, one row per test visit |
| **New key per submission** | `01_dummy`, `02_ridge`, `03_hgbr`, `04_tabular_pipeline`, `05_dataops` — never reuse a key |
