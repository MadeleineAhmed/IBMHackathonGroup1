# 13 — Ensemble sur les features de calibration personnelle

| | |
|---|---|
| **Script** | `experiments/13_ensemble.py` |
| **Hub keys** | `13_ensemble` — [EstimatorReport](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/estimators/44049) (URL Kaggle) · `13_ensemble_cv` — [CV 5 folds](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/cross-validations/44019) |
| **Submission** | `submission_13_ensemble.csv` |
| **Modèle** | même mélange que `11_ensemble` (3 HistGradientBoosting à 30 % + Ridge à 10 %, puis lissage par patient), reconstruit sur `PatientPersonalFeatures` de `12` |

---

## Résultats

| Modèle | RMSE (GroupKFold par patient) |
|---|---|
| 12 — modèle seul | 3.29 ± 0.06 |
| **13 — ensemble** | **3.26 ± 0.06** |

Holdout (EstimatorReport) : 3.33.

Même gain que lors du passage de `10` à `11` (−0.03 à −0.04) : l'ensemble apporte un petit bonus régulier, quelles que soient les features.

---

## L'idée

Identique à `11_ensemble` : plusieurs bons modèles **se trompent un peu différemment**, et leur moyenne compense une partie des erreurs.

- **3 HistGradientBoosting** : les 3 meilleurs réglages trouvés par la recherche de `10` (vitesse d'apprentissage, taille des arbres, part des colonnes vues différentes), 30 % chacun ;
- **1 Ridge** sur les mêmes features, 10 % : plus faible seul, mais d'une autre famille, donc des erreurs différentes ;
- **lissage par patient** (parabole en fonction de l'âge) appliqué une seule fois sur la moyenne.

Les poids n'ont pas été re-optimisés : on reprend ceux de `11` (validés à l'époque par une grille 10/20/30 % et par une optimisation libre), ce qui évite de sur-ajuster les poids aux folds.

## Coût

Chaque membre de l'ensemble recalcule ses features : l'entraînement est ~4× plus long que `12` (quelques minutes pour la CV complète).

## Commandes

```bash
python experiments/13_ensemble.py --no-hub   # test local
python experiments/13_ensemble.py            # push Hub (2 rapports) + submission
```
