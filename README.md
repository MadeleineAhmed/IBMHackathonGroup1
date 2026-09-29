# IBM × Probabl Hackathon — Group 1

> **Competition:** [ibm-probabl-hackathon on Kaggle](https://www.kaggle.com/t/ece2ca6a5b0b456b85692ad66a5aee6d)  
> **Lab guide:** [probabl-ai/hackathon](https://github.com/probabl-ai/hackathon) (`docs/GUIDED.md`, `docs/CONTEXT.md`)  
> **Skore Hub:** workspace [`ibmhackathongroup1`](https://skore.probabl.ai/ibmhackathongroup1) · project [`ibm-hackathon`](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon)  
> **Stack:** Python 3.12+ · skore 0.26 · scikit-learn 1.9 · skrub 0.10

---

## The task

We get a table of Parkinson's disease **patient clinic visits** — one row per visit, many visits per patient. At each visit the clinic measured motor severity on the **MDS-UPDRS scale** (0–132, higher = worse), but those measured scores are **biased**:

- by when the patient last took levodopa (the drug may still be working during the "OFF" exam),
- by examiner subjectivity,
- and by missing data (the uncomfortable OFF exam is often skipped).

The organisers estimated each visit's **debiased "true OFF" score** (`target`) using information we don't get. **Our job is to learn to reproduce that from the messy inputs alone** and predict it for visits of **new patients**. Regression, scored by **RMSE** (lower = better).

Background: [`context/levodopa_domain.md`](context/levodopa_domain.md).

---

## Getting started (each teammate, once)

1. **Accept the Skore Hub invite** to workspace `ibmhackathongroup1` and join the Kaggle team.
2. **Clone this repo** and run the setup script (needs Python 3.12+):

   | OS | Command |
   |---|---|
   | **Windows** | Double-click `setup\windows\setup.bat` |
   | **macOS / Linux** | `bash setup/unix/setup.sh` |

   It creates `.venv/`, installs `requirements.txt` and the local `parkinson` package, installs the lab skills for Bob (`.bob/skills/`), signs you in to Skore Hub (writes `.skore` — never committed) and runs `scripts/env_check.py`. Safe to re-run; `-NoHub` / `--no-hub` skips the Hub step.
3. **Download the data** from the [Kaggle Data tab](https://www.kaggle.com/competitions/ibm-probabl-hackathon/data) and unzip into `data/`:
   `X_train.csv`, `y_train.csv`, `X_test.csv`, `sample_submission.csv` (raw CSVs are not committed).
4. **Open Bob in the repo root** (Bob CLI or Bob IDE) so it picks up `.bob/skills/`. All submission code must be written and run in Bob.

Activate the environment later with `.venv\Scripts\Activate.ps1` (Windows), `source .venv/bin/activate` (macOS/Linux), or run anything via `./run.sh <script.py>`.

**Key columns:** `patient_id`, `Index`, `age`, `age_at_diagnosis`, `sexM`, `ledd`, `on`, `off`, `time_since_intake_on`, `time_since_intake_off`, `gene`, `cohort`.

---

## Results

**Presentation / methodology write-up:** [`METHODOLOGY.md`](METHODOLOGY.md) — the problem, evaluation protocol, data findings, every step with its rationale, why the final model was chosen, and what did not work.

Every step: patient-grouped 5-fold CV (the decision metric), a Skore Hub report, a Kaggle submission with its report URL. Details and explanations (in French) in `experiments/NN_name.md`; index in [`journal/JOURNAL.md`](journal/JOURNAL.md).

| Experiment | What changed | Grouped CV RMSE | Kaggle public |
|---|---|---|---|
| `01_dummy` | predict the mean | 16.48 | 16.42 |
| `02_ridge` | Ridge + missing-value indicators | 8.54 | 8.39 |
| `03_hgbr` | HistGradientBoosting, grouped CV (steps 10–11) | 7.44 | 7.20 |
| `04_tabular_pipeline` | + `cohort`, `gene` (skrub) | 7.43 | 7.17 |
| `05_dataops` | same, skrub DataOps | 7.43 | 7.14 |
| `06_patient_features` | summaries + trend of each patient's other visits | 3.98 | 3.80 |
| `07_pk_correction` | readings corrected for dose timing | 3.75 | 3.66 |
| `08_smoothing` | per-patient quadratic smoothing of predictions | 3.68 | 3.59 |
| `09_patient_curve` | curved per-patient trend + reliability | 3.50 | 3.46 |
| `10_tuning` | tuned model settings (randomized search) | 3.47 | 3.38 |
| **`11_ensemble`** | **3 tuned models + 10 % Ridge** | **3.43** | **3.35** |

The big jump (7.4 → 4.0) comes from using all of a patient's visits: the target is a smooth increasing curve per patient, and each visit's `on`/`off` is a noisy reading of it.

Shared code: `src/parkinson/data.py` (loading, grouped splits, submission writer), `features.py` (patient / timing-correction / curve features), `models.py` (per-patient smoothing).

---

## Modelling plan

The lab guide walks through five models; each is one experiment, one Hub report and one Kaggle submission. Full code templates: [`context/ml_decisions.md`](context/ml_decisions.md).

| Stem / Hub key | Model | Why |
|---|---|---|
| `01_dummy` | Predict the training mean | Floor to beat; proves data → evaluate → Hub → Kaggle works |
| `02_ridge` | Median-impute + Ridge | Is there any linear signal in the numbers? |
| `03_hgbr` | HistGradientBoostingRegressor | Uses missing values as signal; first **patient-grouped** evaluation |
| `04_tabular_pipeline` | skrub `tabular_pipeline` | Adds `gene` / `cohort` text columns |
| `05_dataops` | skrub DataOps | Same, with the grouped split baked into the pipeline |

Steps 1–2 use the guide's default random row split, so their local RMSE is optimistic and not comparable with steps 3+.

**Beyond the guide** — where we try to beat other teams: patient-level features from all of a patient's visits, pharmacokinetic features, monotonic progression, mixed-effects models, missingness patterns, error analysis, ensembling. Ranked list with rationale: [`context/ml_decisions.md` § Beyond the guide](context/ml_decisions.md#beyond-the-guide-where-we-can-beat-other-teams); tracked as Backlog items in [`journal/JOURNAL.md`](journal/JOURNAL.md).

---

## Rules that sink submissions

| Rule | Detail |
|---|---|
| **Group by patient** | Kaggle's test set holds out whole patients. Evaluate with `GroupKFold(n_splits=5)` on `patient_id`, never a random row split. |
| **No id features** | Drop `patient_id` and `Index` before fitting. |
| **Keep missing values** | A missing `off` usually means the exam was skipped — that is signal. Don't blindly median-impute. |
| **Hub URL on every submission** | Push an `EstimatorReport` / `CrossValidationReport` with a **new** key and paste the printed URL into the Kaggle Submission Description, or the submission is invalid. `eda` is reserved for EDA. |
| **Submission format** | CSV with header `Index,target`, one row per test visit. |

Hub push snippet (every modelling script):

```python
from parkinson.hub import load_skore_credentials
from skore import Project, login

cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="ibm-hackathon", mode="hub", workspace=cfg["workspace"])
project.put("01_dummy", report)   # ComparisonReport not accepted: put comp.reports_["name"]
```

skore 0.26 gotchas (and where the lab guide is wrong): [`context/skore_api_reference.md`](context/skore_api_reference.md).

---

## Repo structure

Layout follows the lab's `organize-ml-workspace` skill so Bob finds everything.

```
.
├── README.md
├── journal/
│   ├── JOURNAL.md          # Experiment index: status, EDA summary, history, backlog
│   ├── NN_name.md          # One design note per experiment (created per experiment)
│   └── setup_log.md        # Setup/infrastructure decisions (J-001 … J-007)
├── experiments/            # One `# %%` script per experiment: 01_dummy.py, …
├── src/parkinson/          # Reusable code; hub.py = load_skore_credentials()
├── context/
│   ├── levodopa_domain.md      # Clinical background, dataset schema, pitfalls
│   ├── ml_decisions.md         # Modelling decisions, code templates, strategy
│   └── skore_api_reference.md  # Verified skore 0.26 API + Hub integration
├── data/                   # Kaggle CSVs (local only); EDA outputs committed here
├── tests/test_skore.py     # Smoke tests for the skore API
├── scripts/
│   ├── skore-agent         # Lab script: Hub sign-in, writes .skore
│   └── env_check.py        # Environment sanity check
├── setup/
│   ├── windows/            # setup.bat (double-click) → setup.ps1
│   └── unix/               # setup.sh, activate.fish
├── .bob/skills/            # Lab skills for Bob
├── run.sh                  # Run Python in the project env without activating
├── requirements.txt        # Dependencies (skore / skore-cli / skrub pinned)
└── pyproject.toml          # Makes src/parkinson installable
```

Not in git: `.skore` (Hub API key), `.venv/`, `data/*.csv`, `submission*.csv`, `reports/`, `scratch/`.
