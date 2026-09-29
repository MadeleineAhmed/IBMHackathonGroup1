# Domain Reference: Levodopa & Parkinson's Motor Score Prediction

> Last updated: 2026-09-29  
> Source: probabl-ai/hackathon CONTEXT.md + GUIDED.md, Wikipedia – Levodopa  
> Purpose: Grounding document for the levodopa true-OFF MDS-UPDRS regression task.

---

## 1. What is Levodopa?

Levodopa (L-DOPA) is the **gold-standard pharmacological treatment for Parkinson's disease (PD)**.  
It is a dopamine precursor that crosses the blood–brain barrier (dopamine itself cannot), where it is converted to dopamine by the enzyme aromatic L-amino acid decarboxylase (AAAD/DDC).

- **Formula:** C₉H₁₁NO₄  
- **Bioavailability:** ~30% (oral)  
- **Elimination half-life:** 0.75–1.5 hours — the short half-life drives ON/OFF fluctuations  
- **Primary excretion:** Renal (70–80%)  
- **Always combined with:** carbidopa or benserazide (peripheral DDCI) to prevent peripheral conversion

---

## 2. ON and OFF States — Clinical Definitions

### ON state
- Levodopa is active; dopamine receptors are adequately stimulated.
- Patient has good motor control: reduced rigidity, bradykinesia, tremor.

### OFF state
- Levodopa blood levels below therapeutic threshold.
- Return of PD motor symptoms: bradykinesia, rigidity, tremor, freezing gait.
- **Scientifically important:** the OFF score is the best proxy of underlying neurodegeneration (disease severity).

### Disease progression (from CONTEXT.md)
- **Early / honeymoon phase:** motor control fairly stable through the day.
- **Later:** *wearing-off* — symptoms return at the end of a dose.
- **Advanced:** patients fluctuate between ON and OFF unpredictably.

The therapeutic window narrows over time: peaks hit dyskinesia; troughs hit wearing-off (see Figure 2 in CONTEXT.md).

---

## 3. MDS-UPDRS Motor Score

The **Movement Disorder Society Unified Parkinson's Disease Rating Scale (MDS-UPDRS)** Part III is the clinical standard for measuring motor severity.

- **18 motor items** (akinesia, rigidity, tremor across body parts) → 33 subscores
- **Scale:** 0 (normal) to 4 (severe) per item
- **Total range:** 0 – 132
- Scores typically **improve 50–100%** from OFF to ON state

---

## 4. The Challenge: Predicting the "True OFF" Score

### Why observed scores are biased (from CONTEXT.md)
Observed OFF motor scores are noisy proxies of neurodegeneration because:

1. **Human subjectivity** in scoring
2. **Missing / incorrect values** — visits with irregular data
3. **Fluctuating levodopa blood levels** at the time of assessment
4. **OFF exams are uncomfortable** → often skipped; only ON score available

### The target variable
For this hackathon, a **"true OFF"** score was estimated by removing those biases from the multi-cohort dataset. This clean estimate is not available in real clinical care — the challenge is to **recover that debiasing process** and predict it at every patient visit.

**Target column: `target`** (continuous, range ~0–132, represents the debiased true OFF MDS-UPDRS motor score)

**This is a REGRESSION problem, not classification.**

---

## 5. Dataset Structure

From GUIDED.md — the data is **synthetic**, built to match real multi-cohort PD records.

### Files (download from Kaggle, not in git)
| File | Description |
|---|---|
| `data/X_train.csv` | Feature table for training visits |
| `data/y_train.csv` | Target column (`target`) for training visits |
| `data/X_test.csv` | Feature table for test visits (predict these) |
| `data/sample_submission.csv` | Template: `Index,target` columns |

### Known columns (from GUIDED.md)
| Column | Type | Notes |
|---|---|---|
| `patient_id` | ID | Multiple visits per patient. **Never use as a feature — leakage.** Test patients don't overlap train. |
| `Index` | ID | Row identifier. **Drop before fitting.** |
| `age` | numeric | Patient age at visit |
| `age_at_diagnosis` | numeric | Age when PD was diagnosed |
| `sexM` | binary | Sex (1=male) |
| `ledd` | numeric | Levodopa equivalent daily dose (mg). Often missing. |
| `on` | numeric | Measured ON motor score (biased). Often missing. |
| `off` | numeric | Measured OFF motor score (biased). Often missing. |
| `time_since_intake_on` | numeric | Hours between last levodopa dose and ON measurement. Often missing. |
| `time_since_intake_off` | numeric | Hours between last levodopa dose and OFF measurement. Often missing. |
| `gene` | string | Genetic variant (e.g. "GBA"). Often missing / high cardinality. |
| `cohort` | string | Multi-cohort label. Low cardinality. |
| `target` | numeric | **True OFF score** (train only). This is what we predict. |

### Key structural properties
- Each **row is a visit**; one patient has multiple rows.
- **`X_test` patients do NOT overlap train patients** — the Kaggle holdout is by `patient_id`.
- This means **GroupKFold by `patient_id`** is required for honest CV (random row splits leak patient data).

---

## 6. Missingness as Signal (important!)

From CONTEXT.md and GUIDED.md:
> "Missingness is part of the generative process."

- A **missing `off` value** usually means the visit was ON-only — the uncomfortable OFF exam was skipped.
- This absence *itself* carries information about the patient's state and study protocol.
- **Do NOT blindly impute NaN with median** — this destroys the signal.
- `HistGradientBoostingRegressor` natively handles NaN without imputation (routes NaN values during tree splits as a learnable decision).

---

## 7. Drug Timing as a Key Feature

From CONTEXT.md:
> "Levodopa effect is proportional to blood concentration: a fast absorption phase, then a decline toward zero."

- `time_since_intake_on` and `time_since_intake_off` encode when during the pharmacokinetic curve the measurement was taken.
- A high `time_since_intake_off` means the OFF exam was done long after the last dose → more representative of the true debiased state.
- These timing columns are critical for **unbiasing** the observed scores.

---

## 8. Temporal Progression

- `age - age_at_diagnosis` ≈ `time_since_diagnosis` (years since PD onset)
- Disease duration is a proxy for neurodegeneration severity
- True OFF scores are expected to **increase monotonically with disease duration** (worsening)
- Note: test set may already have a `time_since_diagnosis` column directly

---

## 9. Evaluation Metric

**RMSE (Root Mean Squared Error)** on the `target` column.  
Lower is better. The competition leaderboard ranks by RMSE.

Benchmark / floor: predicting the mean OFF score adjusted for time since disease onset.  
Any real model should beat this floor.

---

## 10. Submission Requirements

Every Kaggle submission MUST have a Skore Hub `EstimatorReport` URL in the Submission Description.  
Submissions without it are **invalid** even if Kaggle scored the CSV.

- CSV format: `Index,target` (one row per test visit, matching `sample_submission.csv`)
- Report must be pushed with a **new** hub project key per submission (e.g. `01_dummy`, `02_ridge`, …)
- Reserved key `eda` must NOT be used for a model report

---

## 11. Levodopa Pharmacokinetics Summary

| Parameter | Value |
|---|---|
| Time to peak plasma (Tmax), oral IR | ~30–60 min |
| Elimination half-life | 0.75–1.5 h |
| Clinical benefit duration (IR) | ~3–5 h |
| Score improvement ON vs OFF | 50–100% |

---

## 12. Modelling Pitfalls (from CONTEXT.md)

| Pitfall | Why it happens | Fix |
|---|---|---|
| Row-level random split | Same patient in train and test fold → memorization | `GroupKFold(patient_id)` |
| Using `Index` or `patient_id` as features | Direct row ID lookup → leakage | Drop before fitting |
| Imputing `off` NaN with median | Destroys missingness-as-signal | Use HGBR (NaN-native) or mark with indicator |
| Ignoring timing columns | Biased raw scores without debiasing | Include `time_since_intake_*` |
| Predicting ON state as proxy | ON score ≠ true OFF; improvement is 50–100% | Predict `target` directly |
