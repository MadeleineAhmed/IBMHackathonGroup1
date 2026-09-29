# 08 — Lissage des prédictions par patient

| | |
|---|---|
| **Script** | `experiments/08_smoothing.py` |
| **Code réutilisable** | `src/parkinson/models.py` — `PatientSmoother` |
| **Hub keys** | `08_smoothing` — [EstimatorReport](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/estimators/42789) (URL Kaggle) · `08_smoothing_cv` — [CV 5 folds](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/cross-validations/42783) |
| **Submission** | `submission_08_smoothing.csv` |
| **Modèle** | `PatientSmoother(degree=2)` autour du modèle `07` |

---

## Résultats

| Modèle | RMSE (GroupKFold par patient) |
|---|---|
| 07 — correction PK | 3.75 ± 0.06 |
| **08 — + lissage quadratique par patient** | **3.68 ± 0.06** |

Holdout (EstimatorReport) : 3.76.

### Variantes comparées (sur les prédictions hors-fold de `07`)

| Lissage | RMSE | moitié lissé / moitié brut |
|---|---|---|
| aucun | 3.748 | — |
| droite | 3.811 | 3.736 |
| **parabole (degré 2)** | **3.679** | 3.695 |
| degré 3 | 3.700 | 3.712 |
| isotonique (croissant) | 3.733 | 3.739 |

La droite dégrade : le vrai `target` d'un patient est légèrement courbé (une parabole l'ajuste à ~0.2 point, une droite à ~1.2). Le degré 3 commence à suivre le bruit.

---

## L'idée

Le vrai `target` d'un patient est une courbe lisse en fonction de l'âge, mais le modèle `07` prédit chaque visite séparément : ses prédictions pour un même patient oscillent d'environ ±0.9 point autour d'une parabole. `PatientSmoother` :

1. entraîne le modèle `07` normalement ;
2. à la prédiction, pour chaque patient, ajuste une **parabole en fonction de l'âge** sur ses prédictions ;
3. renvoie les valeurs de la parabole.

Seuls `patient_id` et `age` sont utilisés à la prédiction. Avec la CV groupée, un patient est entièrement en train ou en validation : pas de fuite.

## Ce qu'il reste

Le gain est modeste (−0.07), comme prévu par l'analyse d'erreur : l'essentiel de l'erreur restante est un **décalage de niveau par patient**, que le lissage ne corrige pas. Autre constat de `06`, toujours présent : les scores élevés sont sous-estimés et les faibles surestimés (tirage vers la moyenne).

## Commandes

```bash
python experiments/08_smoothing.py --no-hub   # test local
python experiments/08_smoothing.py            # push Hub (2 rapports) + submission
```
