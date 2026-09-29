# JOURNAL

<!--
Durable index of every experiment in this workspace. Four sections,
in order: Status, Data understanding (EDA), History, Backlog - keep
them so the file stays quick to scan. Each journal/NN_short_name.md
design note pairs one-to-one with experiments/NN_short_name.py (same
stem). The experiment stem doubles as the Skore Hub key (01_dummy, ...).

Setup / infrastructure history (env, Hub, API corrections) lives in
journal/setup_log.md. Domain and modelling background lives in context/.
-->

## Status

- **Project / dataset:** `ibm-probabl-hackathon` (Kaggle) - regression: predict the debiased "true OFF" MDS-UPDRS motor score (`target`, 0–132) per patient visit
- **Goal:** lowest RMSE on the Kaggle test set (patients disjoint from train); every submission backed by a Skore Hub report URL
- **Last experiment:** `08_smoothing` - done
- **Last result:** 07 + per-patient quadratic smoothing of predictions: RMSE 3.68 ± 0.06 (patient GroupKFold)

- **Workspace decisions** (immutable unless the user pivots):
  - tabular library: pandas - recorded: 2026-09-29
  - env manager: pip+venv (`.venv/`, created by `setup/`); local fallback: pyenv virtualenv `skore` - recorded: 2026-09-29
  - agent feature: installed (skore-cli 0.4.1 via requirements.txt) - recorded: 2026-09-29
  - optional features: none - recorded: 2026-09-29
  - package name (`src/<pkg>/`): parkinson - recorded: 2026-09-29
  - skore mode: hub - recorded: 2026-09-29
  - skore hub workspace: ibmhackathongroup1 (project `ibm-hackathon`) - recorded: 2026-09-29
  - skore mlflow tracking uri: n/a - recorded: 2026-09-29
  - CV splitter family: GroupKFold (groups=`patient_id`) - recorded: 2026-09-29

## Data understanding (EDA)

- **Status:** not started - data must first be downloaded from Kaggle into `data/`
- **Summary:** n/a
- **Report:** [data/eda.md](../data/eda.md) (Hub key `eda`, reserved for EDA only)

## History

| Stem | Intent (one line) | Status | Headline result | Design note |
|---|---|---|---|---|
| `01_dummy` | DummyRegressor(mean) floor, random row holdout | done | RMSE 16.48 · [Hub](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/estimators/42094) | [01_dummy.md](../experiments/01_dummy.md) |
| `02_ridge` | Median impute + missing indicators + Ridge, 8 numeric features | done | RMSE 8.53 (random) / 8.54 (grouped) · [Hub](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/estimators/42552) | [02_ridge.md](../experiments/02_ridge.md) |
| `03_hgbr` | HistGradientBoosting, NaN-native, 8 numeric features; first patient-grouped CV | done | RMSE 7.44 ± 0.14 (grouped CV) · [Hub](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/estimators/42609) · [CV](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/cross-validations/42602) | [03_hgbr.md](../experiments/03_hgbr.md) |
| `04_tabular_pipeline` | skrub tabular_pipeline: + cohort, gene; ids dropped | done | RMSE 7.43 ± 0.13 (grouped CV) · [Hub](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/estimators/42622) · [CV](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/cross-validations/42616) | [04_tabular_pipeline.md](../experiments/04_tabular_pipeline.md) |
| `05_dataops` | skrub DataOps: TableVectorizer + HGBR, GroupKFold on mark_as_X | done | RMSE 7.43 ± 0.13 (grouped CV) · [Hub](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/estimators/42642) · [CV](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/cross-validations/42636) | [05_dataops.md](../experiments/05_dataops.md) |
| `06_patient_features` | PatientFeatures (per-patient summaries + trend of off/on over age) + HGBR | done | RMSE 3.98 ± 0.06 (grouped CV) · [Hub](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/estimators/42677) · [CV](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/cross-validations/42665) | [06_patient_features.md](../experiments/06_patient_features.md) |
| `07_pk_correction` | PatientPKFeatures: ON ratio / OFF bias by dose timing (learned in-fold), weighted patient trend; min_samples_leaf=100 | done | RMSE 3.75 ± 0.06 (grouped CV) · [Hub](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/estimators/42746) · [CV](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/cross-validations/42740) | [07_pk_correction.md](../experiments/07_pk_correction.md) |
| `08_smoothing` | PatientSmoother: per-patient quadratic in age through 07's predictions | done | RMSE 3.68 ± 0.06 (grouped CV) · [Hub](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/estimators/42789) · [CV](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/cross-validations/42783) | [08_smoothing.md](../experiments/08_smoothing.md) |

## Backlog

| # | Item | Source |
|---|---|---|
| B9 | Mixed-effects model (patient random intercept + slope) as a model or stacking input | my-pick |
| B10 | Missingness-pattern indicators (which of on/off/ledd/timing are missing), per visit and per patient | my-pick |
| B11 | Error analysis by cohort / gene / missingness pattern from skore reports, then targeted features | my-pick |
| B12 | Ensemble of best grouped-CV models | my-pick |
