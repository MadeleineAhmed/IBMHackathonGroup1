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

**Teammates: run the setup script once after cloning.** You need Python 3.12+ and must already be a member of the Hub workspace [`ibmhackathongroup1`](https://skore.probabl.ai/ibmhackathongroup1).

| OS | Command |
|---|---|
| **Windows** | Double-click `setup\windows\setup.bat` (or in PowerShell: `powershell -ExecutionPolicy Bypass -File setup\windows\setup.ps1`) |
| **macOS / Linux** | `bash setup/unix/setup.sh` |

The script creates `.venv/`, installs `requirements.txt` plus the local `parkinson` package, installs the lab skills for Bob into `.bob/skills/`, signs you in to Skore Hub (writes `.skore`, never committed) and runs `env_check.py`. Add `-NoHub` (Windows) or `--no-hub` (macOS/Linux) to skip the Hub step. Safe to re-run.

Afterwards:

```bash
# Windows
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate      # or: ./run.sh your_script.py
```

Then download the competition CSVs from Kaggle into `data/`.

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
├── requirements.txt        # Team dependencies (skore / skore-cli / skrub pinned)
├── pyproject.toml          # Declares src/parkinson/ as an installable package
├── setup/
│   ├── windows/setup.bat   # Windows: double-click (runs setup.ps1)
│   └── unix/setup.sh       # macOS / Linux: bash setup/unix/setup.sh
├── scripts/skore-agent     # Lab script: Hub sign-in, writes .skore
├── src/parkinson/
│   └── hub.py              # load_skore_credentials() for Hub pushes
├── .bob/skills/            # Lab skills for Bob (installed by the setup script)
├── context/
│   ├── levodopa_domain.md  # Clinical background, dataset schema, modelling pitfalls
│   ├── skore_api_reference.md  # Correct skore v0.26 API + Hub integration
│   └── ml_decisions.md     # Modelling decisions, feature set, full code templates
├── data/                   # NOT in git — download from Kaggle
└── .skore                  # NOT in git — Hub API key, written by scripts/skore-agent
```

---

## The task — plain terms

You're given a table of Parkinson's disease **patient clinic visits**. Each row is one visit; one patient has many rows over time.

For each visit, the clinic measured motor severity with the **MDS-UPDRS scale** (0–132, higher = worse). But those numbers are biased — they depend on when the patient last took their dose, how subjective the examiner was, and whether they even ran the uncomfortable OFF exam at all.

**Our job:** predict the **true, debiased OFF score** for every visit in the test set. This is what the motor severity *actually is*, stripped of all the measurement noise. The hackathon organisers estimated it by running a debiasing model on real multi-cohort records — we have to reconstruct that process.

---

## The 5-step ladder

The hackathon guides you through an explicit progression. Each step = one Kaggle submission:

| Step | What | Why |
|---|---|---|
| **1 — Dummy** | Predict the training mean for everyone | Proves infra works: data load → eval → Hub push → CSV upload |
| **2 — Ridge** | Linear model with median imputation | Checks whether numbers carry any linear signal |
| **3 — HGBR** | `HistGradientBoostingRegressor` + `GroupKFold` | Handles missing values natively; first honest patient-grouped eval |
| **4 — tabular_pipeline** | `skrub` auto-encoding | Adds `gene`/`cohort` string columns via automatic encoding |
| **5 — DataOps** | `skrub` DataOps graph | Groups baked into the graph — nothing can drift or be forgotten |

---

## Two things that will sink you if you get them wrong

**1. CV leakage** — The Kaggle test set is held out **by patient** (entire patients, not random rows). If you use a random row split for local validation, the same patient appears on both sides, your RMSE looks great, and you bomb on the leaderboard. Always use `GroupKFold(n_splits=5)` with `groups=patient_id`.

**2. Missing Hub URL** — Every Kaggle submission **must** have a Skore Hub `EstimatorReport` URL pasted into the Submission Description field on Kaggle. If it's missing, the row is **invalid** even if Kaggle scored your CSV. This is a hard competition rule.

---

## What to do right now (before any coding)

1. **Join Kaggle** and form a team (max 4 people) — do this first
2. **Create a Skore Hub workspace** at [skore.probabl.ai](https://skore.probabl.ai) — name it **exactly** the same as your Kaggle team name — invite teammates
3. **Clone `probabl-ai/hackathon`** (the official lab repo — has `scripts/skore-agent` and the `parkinson/` package)
4. **Run `python scripts/skore-agent`** from that clone — opens browser, sign in, writes `.skore`
5. **Download the data** from the Kaggle Data tab into `data/`
6. **Start with the dummy regressor** (J-008 in the journal) — don't skip the floor check

---

## Journal

All decisions, corrections, and experiment results are logged in [`JOURNAL.md`](JOURNAL.md) with checklist IDs (J-001, J-002, …).

Completed: J-001 (env setup) · J-002 (skore tests) · J-003 (tooling) · J-004 (context docs) · J-005 (hackathon spec ingestion)
Next: J-006 (Hub auth) → J-007 (data + EDA) → J-008 (dummy baseline) → …
