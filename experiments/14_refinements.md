# 14 — Distance aux vraies mesures OFF + lissage toujours croissant

| | |
|---|---|
| **Script** | `experiments/14_refinements.py` |
| **Code réutilisable** | `src/parkinson/features.py` — `PatientDistanceFeatures` · `src/parkinson/models.py` — `PatientSmoother(rising=True)` |
| **Hub keys** | `14_refinements` — [EstimatorReport](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/estimators/44348) (URL Kaggle) · `14_refinements_cv` — [CV 5 folds](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/cross-validations/44324) |
| **Submission** | `submission_14_refinements.csv` |
| **Modèle** | `PatientSmoother(degree=2, rising=True)` autour de `PatientDistanceFeatures` → HistGradientBoosting (réglages de `10`) |

---

## Résultats

| Modèle | RMSE (GroupKFold par patient) |
|---|---|
| 12 — calibration personnelle | 3.29 ± 0.06 |
| **14 — + distance aux mesures OFF + lissage croissant** | **3.27 ± 0.06** |

Holdout (EstimatorReport) : 3.36 (contre 3.37 pour `12`). Testé d'abord en local (`--no-hub`) ; poussé seulement parce qu'il bat `12` sur les deux mesures.

### Variantes comparées (modèle seul, CV groupée)

| Variante | RMSE |
|---|---|
| `12` (référence) | 3.293 |
| cible en échelle log | 3.337 (pire) |
| cible en racine carrée | 3.307 (pire) |
| lissage forcé à être croissant | 3.285 |
| features « distance aux mesures OFF » | 3.282 |
| **les deux** | **3.275** |

---

## Pourquoi : l'analyse d'erreur de `12`

| Constat sur les prédictions hors-fold de `12` | Valeur |
|---|---|
| Part de l'erreur qui est un décalage de niveau par patient | 62 % — **non corrélé** au nombre de mesures, à la dose, à la durée… |
| RMSE des patients avec 0–1 mesure OFF | ~5 (contre 3.1 avec ≥ 5 mesures) |
| RMSE de la **dernière** visite | 4.07 (contre 3.14 au milieu de l'historique) |
| RMSE selon la sévérité | 1.8 sous 20 points, 9.5 au-dessus de 80 |

## Les deux changements

1. **`PatientDistanceFeatures`** — pour chaque visite : l'écart d'âge avec la visite la plus proche où OFF a été mesuré, l'estimation OFF corrigée à cette visite, et l'écart d'âge avec la dernière visite du patient. Le modèle sait ainsi à quel point une visite est « loin des vraies données ».
2. **Lissage toujours croissant** — après la parabole par patient (`08`), la courbe est projetée sur des valeurs qui ne baissent jamais (le vrai score ne baisse jamais entre deux visites). Aide surtout aux extrémités de l'historique, où une parabole peut redescendre.

## Ce qu'on a appris au passage

- **Pas de forme commune de courbe :** la courbure varie beaucoup d'un patient à l'autre (moyenne −0.05, écart-type 0.71 point/an²), dans les deux sens ; une forme exponentielle ajuste bien moins bien qu'une parabole (2.40 contre 0.33). Chaque patient a vraiment sa propre courbe.
- **Un « biais OFF personnel » n'est pas estimable** à partir des entrées : ON et OFF ne peuvent être calibrés que l'un par rapport à l'autre, sans vérité terrain par patient.
- Les gains deviennent petits (≈ −0.02) : on approche de ce que ces données permettent.

## Commandes

```bash
python experiments/14_refinements.py --no-hub   # test local
python experiments/14_refinements.py            # push Hub (2 rapports) + submission
```
