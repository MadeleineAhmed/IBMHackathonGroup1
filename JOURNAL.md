# Project Journal — Levodopa ON/OFF State Tracking

> **Format:** Each entry has a checklist ID (`J-NNN`), a date, a status badge, a decision/action description, the reasoning behind it, and the source/evidence.  
> **Rule:** Every code action, model decision, or architectural choice gets an entry *before or at the time* it is made. Results are appended to the same entry once known.  
> **Skore sync:** Every experiment entry that produces a report also records the `Project` key used to persist the result.

---

## Legend

| Badge | Meaning |
|---|---|
| ✅ Done | Completed and verified |
| 🔄 In progress | Currently being worked on |
| ⏳ Pending | Planned, not started |
| ❌ Abandoned | Tried, discarded — reason recorded |

---

## Checklist

- [x] J-001 — Project bootstrap & environment setup
- [x] J-002 — Skore preliminary tests
- [x] J-003 — Dev tooling scripts
- [x] J-004 — Context folder & domain grounding
- [x] J-005 — Hackathon spec ingestion & problem reframing (regression, not classification)
- [x] J-006 — Install skore-cli & configure Hub workspace
- [ ] J-007 — Download data from Kaggle & EDA
- [ ] J-008 — Dummy regressor baseline (floor check)
- [ ] J-009 — Ridge linear model
- [ ] J-010 — HistGradientBoostingRegressor (NaN-native)
- [ ] J-011 — skrub tabular_pipeline (mixed types)
- [ ] J-012 — skrub DataOps with grouped CV
- [ ] J-013 — Final model selection & Kaggle submission

---

## Entries

---

### J-001 — Project bootstrap & environment setup
**Date:** 2026-09-29  
**Status:** ✅ Done  
**Category:** Infrastructure

#### What was done
- Created a `pyenv` virtualenv named `skore` running Python 3.14.7.
- Installed `skore==0.26.0` and its full dependency tree (scikit-learn 1.9.1, numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, matplotlib 3.11.2, skrub 0.10.1).
- Identified and resolved a **module shadowing bug**: an empty `skore.py` at the workspace root was being imported instead of the installed library. File was renamed to `test_skore.py`.
- Confirmed all core library imports work from the correct interpreter path:  
  `/home/madeleine/.pyenv/versions/skore/bin/python`

#### Reasoning
`pyenv` virtualenvs are isolated per-project environments. The `skore` virtualenv ensures reproducibility for the hackathon without polluting the system Python. The naming collision was a critical bug — left unchecked, every `import skore` in the project would silently import an empty file.

#### Source
- `pyenv` documentation; standard Python virtualenv best practices.
- Bug discovered via `python -c "import skore; print(skore.__file__)"` returning the workspace file path.

---

### J-002 — Skore preliminary smoke tests
**Date:** 2026-09-29  
**Status:** ✅ Done  
**Category:** Testing / API validation

#### What was done
Wrote and ran [`test_skore.py`](test_skore.py) covering 7 test cases:

| Test | Verifies |
|---|---|
| `test_import_and_version` | Package imports, `__version__` is a non-empty string |
| `test_public_api_exports` | `CrossValidationReport`, `EstimatorReport`, `ComparisonReport`, `TrainTestSplit` all in namespace |
| `test_train_test_split_basic` | `TrainTestSplit.split()` returns generator of index tuples; correct 80/20 sizes |
| `test_estimator_report_classification` | `EstimatorReport` trains LogisticRegression, metrics summary non-empty |
| `test_estimator_report_regression` | `EstimatorReport` trains LinearRegression, metrics summary non-empty |
| `test_cross_validation_report` | `CrossValidationReport` runs 3-fold CV, summary accessible |
| `test_comparison_report` | `ComparisonReport` wraps two reports, combined summary non-empty |

**Result:** 7/7 passing. Runtime ~3.5s.

Run with: `./run.sh -m pytest test_skore.py -v`

#### API corrections discovered vs. documentation
The skore v0.26 API differs from older examples circulating online:

| Older docs say | Actual v0.26 API |
|---|---|
| `tts.split(X, y)` returns `(X_train, X_test, y_train, y_test)` | Returns a **generator** of `(train_idx, test_idx)` index arrays — sklearn splitter protocol |
| `CrossValidationReport(clf, X, y, cv=3)` | Parameter is `splitter=3`, not `cv=` |

These corrections are documented in [`context/skore_api_reference.md`](context/skore_api_reference.md).

#### Reasoning
Preliminary tests before any real modelling ensure the foundation is solid. Discovering the API differences now saves debugging time later. Having tests also means any future skore upgrade breakage is caught immediately.

#### Source
- Live introspection: `inspect.signature(TrainTestSplit.split)`, `inspect.signature(CrossValidationReport.__init__)`
- PyPI skore 0.26.0 release page.

---

### J-003 — Dev tooling scripts
**Date:** 2026-09-29  
**Status:** ✅ Done  
**Category:** Infrastructure / Developer experience

#### What was done
Created three scripts to eliminate environment-related friction on Arch Linux:

| File | Purpose |
|---|---|
| [`activate.fish`](activate.fish) | Sets `PYENV_VERSION=skore` and prepends the virtualenv bin to `$PATH` for the current fish session. Source with `. activate.fish`. |
| [`run.sh`](run.sh) | Resolves the correct Python binary via `pyenv which python` and `exec`s into it. Use instead of activating: `./run.sh script.py`, `./run.sh -m pytest ...`, `./run.sh -c "..."`. |
| [`env_check.py`](env_check.py) | Sanity check — verifies all key imports and runs a live EstimatorReport + CrossValidationReport smoke test. All 15 checks pass. |

#### Reasoning
Arch Linux with `pyenv` + fish shell can produce subtle `$PATH` and shim resolution issues that waste hours. `run.sh` uses `exec` with the absolute binary path, bypassing all shell activation complexity. `env_check.py` gives a one-command go/no-go before starting a session.

#### Source
- `pyenv` internal: `PYENV_VERSION` env var overrides the active version for a single invocation.
- `pyenv which python` returns the absolute path of the currently selected Python binary.

---

### J-004 — Context folder & domain grounding
**Date:** 2026-09-29  
**Status:** ✅ Done  
**Category:** Documentation / Domain knowledge

#### What was done
Created [`context/`](context/) folder with three reference documents:

| File | Contents |
|---|---|
| [`context/levodopa_domain.md`](context/levodopa_domain.md) | Full domain reference: levodopa pharmacology, ON/OFF state clinical definitions, wearing-off phenomenon, measurement methods, ML target variable definition, expected features, pharmacokinetics summary |
| [`context/skore_api_reference.md`](context/skore_api_reference.md) | Quick-reference for all skore classes used in this project, with correct v0.26 signatures and usage patterns |
| [`context/ml_decisions.md`](context/ml_decisions.md) | Modelling decisions log: target variable, splitting strategy, class imbalance plan, initial model shortlist, feature engineering priorities |

#### Key domain facts captured — SUPERSEDED by J-005
- ~~Primary ML metric: recall on OFF class~~ → **corrected in J-005: RMSE regression**
- ~~Binary classification~~ → **corrected in J-005: regression**
- Levodopa half-life: **0.75–1.5 h** — still valid
- OFF = return of PD motor symptoms — still valid as context

#### Reasoning
The team needs shared domain grounding to make consistent modelling decisions. Capturing this upfront prevents drift where different team members/sessions use different assumptions about the target variable or evaluation metric.

#### Source
- Wikipedia – Levodopa (en.wikipedia.org/wiki/Levodopa)
- Riederer et al. (2025) — mechanism of OFF phases in late PD
- Standard PD clinical literature on UPDRS and motor fluctuation measurement

---

### J-005 — Hackathon spec ingestion & problem reframing
**Date:** 2026-09-29
**Status:** ✅ Done
**Category:** Problem definition / Documentation

#### What was done
Read the full hackathon repository at `https://github.com/probabl-ai/hackathon`:
- `docs/GUIDED.md` — step-by-step lab guide (14 steps)
- `docs/CONTEXT.md` — clinical background, challenge goals, data description, benchmark
- `scripts/skore-agent` — Hub authentication script
- `.gitignore` — confirms `data/*.csv` not in git (must download from Kaggle)

**Critical correction:** the problem is **REGRESSION**, not binary classification as initially assumed.

| Assumption before J-005 | Corrected reality |
|---|---|
| Binary classification (ON vs OFF) | Regression: predict continuous MDS-UPDRS true-OFF score |
| Metric: recall on OFF class | Metric: **RMSE** (lower = better) |
| StratifiedKFold | **GroupKFold(patient_id)** — test holdout is by patient, not row |
| Data source: fake/custom | Data source: Kaggle competition `ibm-probabl-hackathon` |
| `skore.CrossValidationReport(cv=)` | `skore.evaluate()` with `splitter=cv_splits` is the hackathon idiom |

#### Key facts from spec
- **Target:** `target` column in `y_train.csv` — debiased "true OFF" MDS-UPDRS motor score (0–132 continuous)
- **Data:** synthetic multi-cohort, each row is a patient visit, multiple rows per patient
- **Leakage trap:** `patient_id` must be used as `groups` in `GroupKFold` — NOT as a feature
- **Missingness is signal:** `off` NaN usually means exam was skipped (ON-only visit)
- **Key features:** `time_since_intake_on/off` (PK debiasing), `ledd`, `on`, `off`, `age`, `age_at_diagnosis`, `gene`, `cohort`
- **Derived feature:** `time_since_diagnosis = age - age_at_diagnosis`
- **Hub requirement:** every Kaggle submission must include a Skore Hub `EstimatorReport` URL in the Description
- **Hub key convention:** `01_dummy`, `02_ridge`, `03_hgbr`, `04_tabular_pipeline`, `05_dataops` — new key per submission
- **Reserved key:** `eda` — do NOT use for model reports
- **Submission format:** CSV with `Index,target` columns matching `sample_submission.csv`

#### Files updated
- [`context/levodopa_domain.md`](context/levodopa_domain.md) — fully rewritten with correct problem spec
- [`context/ml_decisions.md`](context/ml_decisions.md) — rewritten for regression with 5-step model progression
- [`JOURNAL.md`](JOURNAL.md) checklist — updated to reflect regression task pipeline

#### Reasoning
The hackathon spec completely redefines the problem. All prior context docs assumed binary classification based on the general description of levodopa ON/OFF states. The actual competition is a regression debiasing task. Correcting this early prevents wasted modelling work.

#### Source
- `github.com/probabl-ai/hackathon` — `docs/GUIDED.md`, `docs/CONTEXT.md`
- Kaggle competition: `kaggle.com/t/ece2ca6a5b0b456b85692ad66a5aee6d`

---

### J-006 — Install skore-cli & configure Hub workspace
**Date:** 2026-09-29
**Status:** ✅ Done
**Category:** Infrastructure

#### What was done
- Installed `skore-cli` 0.4.1 and the lab skills (`skore skills install all --repo probabl-ai/skills-hackathon --agent bob`, release 0.1.0 → `.bob/skills/`, 14 skills).
- Created Hub workspace [`ibmhackathongroup1`](https://skore.probabl.ai/ibmhackathongroup1) (matches Kaggle team name) and invited teammates.
- Copied the lab's `scripts/skore-agent` into this repo so `.skore` is written at this repo's root; ran it → `.skore` (gitignored, mode 600).
- Scaffolded the local `parkinson` package (`pyproject.toml`, `src/parkinson/__init__.py`, `src/parkinson/hub.py` = skill template `src_hub.py`), installed editable, so `from parkinson.hub import load_skore_credentials` works as in GUIDED.md.
- Verified hub login via API key and created project [`ibm-hackathon`](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon) (empty).
- Added team setup scripts: `setup/windows/setup.bat` (+ `setup.ps1`) and `setup/unix/setup.sh`, with `requirements.txt` (skore / skore-cli / skrub pinned) and `.gitattributes` for line endings. Unix script verified end-to-end on a clean copy; Windows script not yet run on Windows.

#### API corrections found while verifying docs
| Doc / guide said | Actual skore 0.26 |
|---|---|
| `project.put(key, comparison_report)` | `TypeError` — only `EstimatorReport` / `CrossValidationReport`; push `comp.reports_["name"]` |
| `compare([est1, est2], X, y)` | `compare()` takes reports, not estimators |
| `project.list_item_keys()` | Does not exist; use `project.summarize()` |
| GUIDED.md step 13 `evaluate(pred)` | `ValueError`; use `evaluate(pred.skb.make_learner(), data={"visits": visits})` |
| Skill doc project name `ibm-hackaton` | Use `ibm-hackathon` |

#### Source
- `inspect.getsource(skore.Project.put)`, `skore.evaluate` docstring; synthetic-data run of the DataOps pattern.

---

### J-007 — Download data from Kaggle & EDA
**Date:** —
**Status:** ⏳ Pending
**Category:** Data

#### Plan
1. Download from Kaggle Data tab → unzip into `data/`
2. Verify: `X_train.csv`, `y_train.csv`, `X_test.csv`, `sample_submission.csv`
3. EDA: shape, dtypes, missingness per column, target distribution, patient visit counts
4. Compute `time_since_diagnosis = age - age_at_diagnosis`
5. Check if `X_test` already has `time_since_diagnosis` directly

Skore Hub key: `eda` (reserved for EDA only)

---

### J-008 — Dummy regressor baseline (floor check)
**Date:** —
**Status:** ⏳ Pending
**Category:** Modelling

#### Plan
```python
from sklearn.dummy import DummyRegressor
from skore import evaluate

feature_cols = ["sexM", "age_at_diagnosis", "age", "ledd",
                "time_since_intake_on", "time_since_intake_off", "on", "off"]
X = visits[feature_cols]
y = visits["target"]

dummy = DummyRegressor(strategy="mean")
report = evaluate(dummy, X, y)
report.metrics.rmse()
project.put("01_dummy", report)
```
**Success criterion:** data loads, evaluate runs, RMSE prints, Hub push succeeds. If this fails, fix infra before modelling.

---

### J-009 — Ridge linear model
**Date:** —
**Status:** ⏳ Pending
**Category:** Modelling

#### Plan
```python
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

ridge = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0))
comp = evaluate({"dummy": dummy, "ridge": ridge}, X, y)   # ComparisonReport
project.put("02_ridge", comp.reports_["ridge"])            # put() rejects ComparisonReport
```
Try `alpha` in [0.1, 1.0, 10.0]. Metric: RMSE.

---

### J-010 — HistGradientBoostingRegressor (NaN-native, GroupKFold)
**Date:** —
**Status:** ⏳ Pending
**Category:** Modelling

#### Plan
```python
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

groups = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))
hgbr = HistGradientBoostingRegressor(random_state=0)
report = evaluate(hgbr, X, y, splitter=cv_splits)
project.put("03_hgbr", report)
```
Do NOT impute — NaN is signal. First model with proper patient-grouped CV.

---

### J-011 — skrub tabular_pipeline (mixed types: gene/cohort)
**Date:** —
**Status:** ⏳ Pending
**Category:** Modelling

#### Plan
```python
from skrub import tabular_pipeline

X_full = visits.drop(columns=["Index", "patient_id", "target"])
model = tabular_pipeline("regressor")
report = evaluate(model, X_full, y, splitter=cv_splits)
project.put("04_tabular_pipeline", report)
```
First model to use `gene` and `cohort` string columns via `TableVectorizer`.

---

### J-012 — skrub DataOps with grouped CV baked in
**Date:** —
**Status:** ⏳ Pending
**Category:** Modelling

#### Plan
```python
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

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
project.put("05_dataops", report)
```

---

### J-013 — Final model selection & Kaggle submission
**Date:** —
**Status:** ⏳ Pending
**Category:** Submission

#### Plan
- Select best model by grouped-CV RMSE (J-010/J-011/J-012)
- Fit on **all** training visits
- Predict `X_test`, write `submission.csv` with `Index,target`
- Push final report to Hub with new key
- Upload CSV to Kaggle with Hub URL in Submission Description

#### Submission checklist
- [ ] CSV has columns `Index,target`
- [ ] Rows match `sample_submission.csv`
- [ ] Hub report URL in Kaggle Submission Description
- [ ] Code written and run in Bob
- [ ] Hub key is new (not reused from a previous submission)
