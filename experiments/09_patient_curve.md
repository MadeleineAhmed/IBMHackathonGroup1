# 09 — Tendance courbe par patient

| | |
|---|---|
| **Script** | `experiments/09_patient_curve.py` |
| **Code réutilisable** | `src/parkinson/features.py` — `PatientCurveFeatures` |
| **Hub keys** | `09_patient_curve` — [EstimatorReport](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/estimators/42885) (URL Kaggle) · `09_patient_curve_cv` — [CV 5 folds](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/cross-validations/42878) |
| **Submission** | `submission_09_patient_curve.csv` |
| **Modèle** | `PatientSmoother(degree=2)` autour de `PatientCurveFeatures(min_obs=4)` → HistGradientBoosting (mêmes réglages que `07`) |

---

## Résultats

| Modèle | RMSE (GroupKFold par patient) |
|---|---|
| 08 — lissage | 3.68 ± 0.06 |
| **09 — + tendance courbe par patient** | **3.50 ± 0.08** |

Holdout (EstimatorReport) : 3.62.

### Variantes comparées

| Variante | RMSE |
|---|---|
| **parabole non pondérée, ≥ 4 estimations** | **3.497** |
| parabole pondérée (inverse de la variance) | 3.533 |
| pondérée, ≥ 3 estimations | 3.533 |
| pondérée + paraboles séparées pour ON et OFF | 3.537 |

---

## Pourquoi

Après `08`, l'erreur restante est surtout un **décalage de niveau par patient**. Or les features de tendance de `06`/`07` étaient des **droites**, alors que le vrai `target` d'un patient est légèrement courbé (une parabole l'ajuste à ~0.2 point, une droite à ~1.2). Et le modèle ne savait pas à quel point la tendance d'un patient était fiable.

## Features ajoutées (`PatientCurveFeatures`)

À partir des estimations corrigées `est` de `07` (lectures ON/OFF converties en estimations du `target`) :

| Feature | Sens |
|---|---|
| `p_est_quad` | parabole de `est` en fonction de l'âge, par patient (≥ 4 estimations), évaluée à l'âge de la visite |
| `p_est_curv` | courbure de cette parabole |
| `p_est_resid_sd` | dispersion de `est` autour de la droite du patient : **à quel point ses mesures sont bruitées**, donc à quel point faire confiance à sa tendance |
| `n_obs_est` | nombre de visites avec au moins une mesure |

Comme pour `07`, les corrections sont apprises dans chaque fold d'entraînement ; les nouvelles features ne dépendent que des colonnes d'entrée.

La pondération par la fiabilité des mesures fait un peu moins bien : avec seulement 4 à 12 points par patient, les poids ajoutent de l'instabilité à la parabole.

## Commandes

```bash
python experiments/09_patient_curve.py --no-hub   # test local
python experiments/09_patient_curve.py            # push Hub (2 rapports) + submission
```
