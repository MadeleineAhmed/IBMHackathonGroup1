# 11 — Ensemble de modèles

| | |
|---|---|
| **Script** | `experiments/11_ensemble.py` |
| **Hub keys** | `11_ensemble` — [EstimatorReport](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/estimators/43438) (URL Kaggle) · `11_ensemble_cv` — [CV 5 folds](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/cross-validations/43426) |
| **Submission** | `submission_11_ensemble.csv` |
| **Modèle** | `PatientSmoother(degree=2)` autour d'un `VotingRegressor` : 3 HistGradientBoosting (30 % chacun) + 1 Ridge (10 %), tous sur `PatientCurveFeatures` |

---

## Résultats

| Modèle | RMSE (GroupKFold par patient) |
|---|---|
| 10 — meilleur modèle seul | 3.47 ± 0.08 |
| **11 — ensemble** | **3.43 ± 0.07** |

Holdout (EstimatorReport) : 3.53.

### Combinaisons comparées (prédictions hors-fold)

| Combinaison | RMSE |
|---|---|
| HGBR #1 seul | 3.467 |
| HGBR #2 seul | 3.468 |
| HGBR #3 seul | 3.469 |
| Ridge seul (mêmes features) | 3.990 |
| moyenne des 3 HGBR | 3.437 |
| **moyenne des 3 HGBR + 10 % Ridge** | **3.430** |
| moyenne des 3 HGBR + 20 % Ridge | 3.438 |
| moyenne des 3 HGBR + 30 % Ridge | 3.461 |
| poids optimisés (moindres carrés positifs, optimiste) | 3.430 — Ridge ≈ 10 % |

---

## L'idée

Plusieurs bons modèles **se trompent un peu différemment** ; en faisant la moyenne, une partie de leurs erreurs se compense.

- **Les 3 HGBR** : les 3 meilleurs réglages de la recherche de `10`. Scores quasi identiques seuls (3.467–3.469), mais construits différemment (vitesse d'apprentissage, taille des arbres, part des colonnes vues) → leur moyenne gagne 0.03.
- **Le Ridge** : un modèle linéaire, bien plus faible seul (3.99), mais d'une autre famille. Ses erreurs ne sont corrélées qu'à 0.81 avec celles des arbres : à petite dose (10 %), il corrige certaines erreurs des arbres. Au-delà de 20 %, sa faiblesse l'emporte.
- **Le poids de 10 %** a été choisi sur la grille 10/20/30 % ; une optimisation libre des poids (NNLS) retombe sur ~10 % pour le Ridge et ~⅓ par HGBR, ce qui confirme le choix sans risquer de sur-ajuster les poids.
- **Le lissage par patient** est appliqué une seule fois, sur la moyenne : ajuster une parabole est une opération linéaire, donc lisser la moyenne revient à moyenner les versions lissées.

## Précautions

- Les réglages et les poids ont été choisis sur les mêmes folds que ceux de l'évaluation : 3.43 est légèrement optimiste. Le score Kaggle sert de contrôle indépendant.
- Le modèle est 4× plus long à entraîner (chaque membre recalcule ses features) : ~2 min pour la CV complète.

## Commandes

```bash
python experiments/11_ensemble.py --no-hub   # test local
python experiments/11_ensemble.py            # push Hub (2 rapports) + submission
```
