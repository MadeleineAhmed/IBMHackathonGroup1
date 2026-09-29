# Étape 12 — skrub `tabular_pipeline` : types mixtes sans encodage manuel

| | |
|---|---|
| **Script** | `experiments/04_tabular_pipeline.py` |
| **Hub keys** | `04_tabular_pipeline` — [EstimatorReport](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/estimators/42622) (URL Kaggle) · `04_tabular_pipeline_cv` — [CV 5 folds](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/cross-validations/42616) |
| **Submission** | `submission_04_tabular_pipeline.csv` |
| **Features** | toutes les colonnes sauf `Index`, `patient_id`, `target` : les 8 numériques + `cohort` + `gene` |

---

## Résultats

| Modèle | RMSE (GroupKFold par patient) |
|---|---|
| Ridge + indicateurs | 8.54 |
| HistGradientBoosting, 8 numériques | 7.44 ± 0.14 |
| **tabular_pipeline (+ `cohort`, `gene`)** | **7.43 ± 0.13** |

Holdout (EstimatorReport, ~20 % des patients) : 7.45.

**Gain quasi nul.** C'est cohérent avec l'exploration des données : le rythme de progression d'un patient ne dépend presque pas de son gène ni de sa cohorte (pente moyenne ≈ 2.4–2.7 points/an dans tous les groupes). Ces colonnes sont aussi constantes pour un patient donné : elles décrivent le patient, pas la visite.

---

## Comment ça marche

`tabular_pipeline("regressor")` enchaîne deux étapes :

1. **`TableVectorizer`** choisit un encodeur par colonne selon sa cardinalité (nombre de valeurs distinctes) :
   - `cohort` (2 valeurs : A, B) et `gene` (4 valeurs + manquant) → peu de valeurs → encodage adapté aux catégories ;
   - colonnes numériques → passées telles quelles, `NaN` compris.
2. **`HistGradientBoostingRegressor`** fait la prédiction (mêmes propriétés qu'à l'étape 11 : non linéaire, `NaN` natifs).

`Index` et `patient_id` sont retirés avant l'entraînement : ce sont des identifiants, pas des variables cliniques. Les garder permettrait au modèle de mémoriser des identifiants (fuite).

## Ce qu'on en retient

Le plafond vers ~7.4 ne vient pas des colonnes manquantes : **chaque visite est encore prédite isolément**. Or le `target` d'un patient suit une courbe lisse sur ses 4 à 12 visites. Le prochain gros gain viendra des features qui regroupent les visites d'un même patient (`06_patient_features`).

## Commandes

```bash
python experiments/04_tabular_pipeline.py --no-hub   # test local
python experiments/04_tabular_pipeline.py            # push Hub (2 rapports) + submission
```
