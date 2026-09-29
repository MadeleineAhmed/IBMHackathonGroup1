# IBM × Probabl Hackathon — Group 1

> **Competition:** [ibm-probabl-hackathon on Kaggle](https://www.kaggle.com/t/ece2ca6a5b0b456b85692ad66a5aee6d)  
> **Hackathon repo:** [probabl-ai/hackathon](https://github.com/probabl-ai/hackathon)  
> **Stack:** Python 3.14 · skore 0.26 · scikit-learn 1.9 · skrub 0.10

---

## What we are building

Parkinson's disease patients on levodopa therapy fluctuate between an **ON state** (medication working) and an **OFF state** (medication worn off). The OFF motor score is the best clinical proxy of neurodegeneration, but scores measured in the clinic are **biased** by:

- Human scoring subjectivity
- The time of the levodopa dose relative to the assessment
- Missing data (OFF exams are uncomfortable and often skipped)

**Our task:** given a table of patient visits (demographics, dose, timing, observed ON/OFF scores), predict the **debiased "true OFF" MDS-UPDRS motor score** (`target`) for each visit.

This is a **regression problem** scored by **RMSE** (lower = better).

---

## Data

Download from the [Kaggle Data tab](https://www.kaggle.com/competitions/ibm-probabl-hackathon/data) and unzip into `data/`. Files are not committed to git.

| File | Description |
|---|---|
| `data/X_train.csv` | Features for training visits |
| `data/y_train.csv` | True-OFF target scores for training visits |
| `data/X_test.csv` | Features for test visits (predict these) |
| `data/sample_submission.csv` | Submission template: `Index,target` |

**Key columns:** `patient_id`, `age`, `age_at_diagnosis`, `sexM`, `ledd`, `on`, `off`, `time_since_intake_on`, `time_since_intake_off`, `gene`, `cohort`

**Critical:** `X_test` patients do **not** overlap train — the holdout is by patient.  Use `GroupKFold(patient_id)` for CV, never a random row split.

---

## Environment setup

```bash
# Activate the skore pyenv virtualenv (fish shell)
source activate.fish

# Or run any script without activating:
./run.sh your_script.py
./run.sh -m pytest test_skore.py -v
./run.sh env_check.py   # full environment sanity check
```

Requirements are managed via `pyenv virtualenv skore` (Python 3.14.7).  
All dependencies are already installed — no `pip install` needed for core work.

---

## Skore Hub setup (required for Kaggle submissions)

Every Kaggle submission must include a Skore Hub `EstimatorReport` URL in the Submission Description. Invalid otherwise.

```bash
# One-time setup per machine — from the probabl-ai/hackathon repo root:
python scripts/skore-agent
# Opens browser → sign in at https://skore.probabl.ai → writes .skore
```

Hub push pattern (in every modelling script):
```python
from parkinson.hub import load_skore_credentials
from skore import Project, login

cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="ibm-hackathon", mode="hub", workspace=cfg["workspace"])
project.put("01_dummy", report)
# Copy the printed URL into the Kaggle Submission Description
```

---

## Modelling plan (5 steps)

| Step | Model | Hub key | Notes |
|---|---|---|---|
| 1 | `DummyRegressor(strategy="mean")` | `01_dummy` | Floor — if you can't beat this, infra is broken |
| 2 | `Ridge` + median imputation | `02_ridge` | Linear baseline with numeric features |
| 3 | `HistGradientBoostingRegressor` | `03_hgbr` | NaN-native; first run with `GroupKFold` |
| 4 | `skrub.tabular_pipeline("regressor")` | `04_tabular_pipeline` | Adds `gene`/`cohort` string features |
| 5 | `skrub` DataOps graph | `05_dataops` | Groups baked into the graph — no drift risk |

See [`context/ml_decisions.md`](context/ml_decisions.md) for full code templates.

---

## Key pitfalls to avoid

| Pitfall | Fix |
|---|---|
| Random row CV split | Use `GroupKFold(patient_id)` — same patient must not appear on both sides |
| Using `patient_id` or `Index` as features | Drop them before fitting |
| Imputing `off` NaN with median | Missingness = exam was skipped = signal; use HGBR natively |
| Reusing a Hub key for a new submission | Always use a new key (`01_`, `02_`, …) |
| Submitting without a Hub URL | The Kaggle submission will be invalid |

---

## Repo structure

```
.
├── JOURNAL.md              # Audit log — every decision recorded with ID, reasoning, source
├── README.md               # This file
├── activate.fish           # Source to activate skore virtualenv in fish
├── run.sh                  # Run any Python script/command without activating
├── env_check.py            # Environment sanity check (all imports + smoke test)
├── test_skore.py           # 7 smoke tests for skore v0.26 API
├── context/
│   ├── levodopa_domain.md  # Clinical background, dataset schema, modelling pitfalls
│   ├── skore_api_reference.md  # Correct skore v0.26 API + Hub integration
│   └── ml_decisions.md     # Modelling decisions, feature set, full code templates
└── data/                   # NOT in git — download from Kaggle
```

---

## Journal

All decisions, corrections, and experiment results are logged in [`JOURNAL.md`](JOURNAL.md) with checklist IDs (J-001, J-002, …).

Completed: J-001 (env setup) · J-002 (skore tests) · J-003 (tooling) · J-004 (context docs) · J-005 (hackathon spec ingestion)  
Next: J-006 (Hub auth) → J-007 (data + EDA) → J-008 (dummy baseline) → …
