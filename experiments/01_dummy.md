# Étape 7 — Dummy : le plancher

| | |
|---|---|
| **Script** | `experiments/01_dummy.py` |
| **Hub key** | `01_dummy` — [rapport](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/estimators/42094) |
| **Submission** | `submission_01_dummy.csv` |
| **Features** | aucune utilisée (le modèle les ignore) |

---

## Résultat

| Modèle | RMSE (split aléatoire 80/20) |
|---|---|
| Dummy (prédit toujours la moyenne du train = 37.47) | **16.48** |

Le RMSE du dummy est égal à l'écart-type du `target` : c'est l'erreur qu'on fait « sans rien savoir ». Tout modèle doit faire mieux.

## Pourquoi cette étape

- Vérifier toute la chaîne : lecture des CSV → `skore.evaluate` → push Hub → CSV de submission → Kaggle.
- Fixer la **référence** : le Hub compare automatiquement chaque modèle à un dummy (check SKD002 « Potential underfitting »). Sur ce rapport, SKD002 est en alerte, ce qui est normal : le modèle *est* le dummy.

## Commandes

```bash
python experiments/01_dummy.py --no-hub   # test local
python experiments/01_dummy.py            # push Hub + submission
```
