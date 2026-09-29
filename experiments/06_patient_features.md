# 06 — Features par patient (au-delà du guide)

| | |
|---|---|
| **Script** | `experiments/06_patient_features.py` |
| **Code réutilisable** | `src/parkinson/features.py` — `PatientFeatures` |
| **Hub keys** | `06_patient_features` — [EstimatorReport](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/estimators/42677) (URL Kaggle) · `06_patient_features_cv` — [CV 5 folds](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/cross-validations/42665) |
| **Submission** | `submission_06_patient_features.csv` |
| **Modèle** | `PatientFeatures` → HistGradientBoosting (`max_iter=500`, `learning_rate=0.05`) |

---

## Résultats

| Modèle | RMSE (GroupKFold par patient) |
|---|---|
| DataOps (étape 13) | 7.43 ± 0.13 |
| **06 — features par patient** | **3.98 ± 0.06** |

Holdout (EstimatorReport) : 4.02. **−46 % d'erreur** par rapport à la fin du guide.

---

## L'idée

L'exploration des données a montré que :

- le `target` d'un patient suit une **courbe lisse et toujours croissante** avec l'âge (une droite par patient l'approche à ~1 point près ; pente médiane +2.4 points/an) ;
- 80 % de la variance du `target` est **entre patients**, pas entre visites ;
- `off` ≈ `target` − 6 et `on` ≈ `target` × 0.45–0.75 : deux mesures **bruitées** (±8–10 points) de cette courbe.

Une visite seule est donc très bruitée, mais les 4 à 12 visites d'un patient, ensemble, disent beaucoup. Les étapes 10–13 prédisaient chaque visite isolément ; ici, chaque visite reçoit aussi un résumé des **autres visites du même patient**.

## Les features ajoutées (`PatientFeatures`)

| Groupe | Features |
|---|---|
| Visite | les 8 numériques, `dur = age − age_at_diagnosis`, `cohort_B`, `gene_code` |
| Position dans l'historique | nombre de visites, rang de la visite, âge moyen du patient, âge centré, étendue des âges |
| Résumés patient | pour `off`, `on`, `ledd` : moyenne, médiane, min, max, nombre de mesures |
| Tendance patient | droite des moindres carrés de `off` et de `on` en fonction de l'âge, **évaluée à l'âge de la visite**, et sa pente |

**Pas de fuite :** tout est calculé à partir des colonnes d'entrée, jamais du `target`, et par patient. Le transformer n'apprend rien (`fit` ne fait que noter les colonnes) : il est donc sûr dans la CV groupée, et valable sur `X_test` où chaque patient a aussi 4 à 12 visites.

## Checks skore sur le rapport

| Check | Résultat | Piste |
|---|---|---|
| SKD001 Overfitting | ⚠️ écart train/test marqué | régulariser (feuilles plus grosses, L2, early stopping) |
| SKD008 Features corrélées | ⚠️ 2 paires > 0.9 (ex. moyenne / médiane) | sans gravité pour des arbres |
| SKD009 vs baseline HGBR | ✅ bien meilleur (baseline 7.6 sur ce holdout) | — |
| SKD012 Features inutiles | 💡 `cohort`, `gene`, `sexM`, `time_since_intake_off` | à retirer ou remplacer |

## Suite

- `07` : corriger chaque mesure de son biais lié à l'heure de la dose avant de faire la tendance patient, et régulariser (pistes SKD001).
- `08` : lisser les prédictions de chaque patient en une courbe croissante.

## Commandes

```bash
python experiments/06_patient_features.py --no-hub   # test local
python experiments/06_patient_features.py            # push Hub (2 rapports) + submission
```
