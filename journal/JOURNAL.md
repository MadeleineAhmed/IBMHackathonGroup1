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
- **Last experiment:** none yet - next is EDA, then `01_dummy`
- **Last result:** n/a

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

## Backlog

| # | Item | Source |
|---|---|---|
| B1 | `01_dummy` - DummyRegressor(mean) floor; default random row holdout as in GUIDED.md step 7 | user (GUIDED.md) |
| B2 | `02_ridge` - median-impute + Ridge on numeric features; tune alpha; push `comp.reports_["ridge"]` if comparing | user (GUIDED.md) |
| B3 | `03_hgbr` - HistGradientBoostingRegressor, NaN-native, numeric features, `splitter=cv_splits` (GroupKFold) | user (GUIDED.md) |
| B4 | `04_tabular_pipeline` - skrub tabular_pipeline with gene/cohort, ids dropped, grouped CV | user (GUIDED.md) |
| B5 | `05_dataops` - skrub DataOps with GroupKFold on mark_as_X; `evaluate(learner, data={"visits": visits})` | user (GUIDED.md) |
| B6 | Patient-level aggregate features from X only (per-patient mean/min/max of on/off, slope vs disease duration, n visits, visit position) | my-pick (context/ml_decisions.md § Beyond the guide) |
| B7 | Pharmacokinetic features: residual-drug proxy from time_since_intake_off, on/off gap and ratio, ledd-adjusted | my-pick (context/levodopa_domain.md § 7, § 11) |
| B8 | Monotonic constraint on disease duration (`monotonic_cst`) and/or per-patient smoothing of predictions | my-pick |
| B9 | Mixed-effects model (patient random intercept + slope) as a model or stacking input | my-pick |
| B10 | Missingness-pattern indicators (which of on/off/ledd/timing are missing), per visit and per patient | my-pick |
| B11 | Error analysis by cohort / gene / missingness pattern from skore reports, then targeted features | my-pick |
| B12 | Ensemble of best grouped-CV models | my-pick |
