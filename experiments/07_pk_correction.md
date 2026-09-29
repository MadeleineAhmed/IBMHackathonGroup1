# 07 — Correction pharmacocinétique des mesures

| | |
|---|---|
| **Script** | `experiments/07_pk_correction.py` |
| **Code réutilisable** | `src/parkinson/features.py` — `PatientPKFeatures` |
| **Hub keys** | `07_pk_correction` — [EstimatorReport](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/estimators/42746) (URL Kaggle) · `07_pk_correction_cv` — [CV 5 folds](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/cross-validations/42740) |
| **Submission** | `submission_07_pk_correction.csv` |
| **Modèle** | `PatientPKFeatures` → HistGradientBoosting (`max_iter=500`, `learning_rate=0.05`, `min_samples_leaf=100`) |

---

## Résultats

| Modèle | RMSE (GroupKFold par patient) |
|---|---|
| 06 — features par patient | 3.98 ± 0.06 |
| 07 — + correction PK | 3.77 |
| **07 — + correction PK + `min_samples_leaf=100`** | **3.75 ± 0.06** |

Holdout (EstimatorReport) : 3.83.

---

## Pourquoi : l'analyse d'erreur de `06`

| Constat sur les prédictions hors-fold de `06` | Valeur |
|---|---|
| Part de l'erreur qui est un **décalage par patient** (toute la courbe trop haute/basse) | 62 % |
| RMSE des patients avec 0–1 mesure OFF | 5.7 |
| RMSE des patients avec ≥ 5 mesures OFF | 3.7 |
| RMSE sur la **dernière** visite d'un patient | 4.6 (vs 3.8 pour la première) |

Les patients sans OFF reposent sur leurs mesures ON, or le score ON dépend fortement de l'heure de la dernière dose. `06` ne corrigeait pas ce biais.

## Comment ça marche (`PatientPKFeatures`)

Appris dans `fit`, **uniquement sur le fold d'entraînement** (le transformer reçoit `y`) :

1. **Courbe ON** : médiane de `on / target` par tranche d'heures depuis la prise (≈ 0.75 dans la première demi-heure → ≈ 0.45 après 1h30 : le médicament agit encore puis s'estompe).
2. **Biais OFF** : moyenne de `off − target` par tranche d'heures depuis la prise (≈ −6 à −14 points : un OFF mesuré trop tôt après la dose est sous-estimé).
3. **Bruit de chaque mesure corrigée** (variance résiduelle) → poids inverses de la variance.

Puis, pour chaque visite (train comme test) :

- `est_on = on / ratio(t_on)` et `est_off = off − biais(t_off)` : deux estimations du `target` ;
- `est` : leur moyenne pondérée (OFF, moins bruité, pèse plus) ;
- par patient : **droite pondérée de `est` en fonction de l'âge**, évaluée à l'âge de la visite, avec sa pente, son poids total et la moyenne de `est`.

Comme les corrections sont apprises dans le pipeline, `skore.evaluate` les réapprend dans chaque fold : **pas de fuite** du `target` de validation.

## Checks skore

| Check | Résultat |
|---|---|
| SKD001 Overfitting | ⚠️ toujours signalé : écart train/test attendu avec 44 000 visites et des features par patient ; la régularisation a amélioré la CV |
| SKD009 vs baseline HGBR | ✅ 3.8 contre 7.6 pour la baseline sur ce holdout |
| SKD012 Features inutiles | 💡 `cohort`, `gene`, `sexM` (plus `time_since_intake_off`, désormais utilisé via la correction) |

## Suite

`08` : les prédictions d'un patient forment encore une courbe un peu irrégulière (±0.9 autour d'une parabole) alors que le vrai `target` est lisse → lissage par patient.

## Commandes

```bash
python experiments/07_pk_correction.py --no-hub   # test local
python experiments/07_pk_correction.py            # push Hub (2 rapports) + submission
```
