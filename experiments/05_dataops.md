# Étape 13 — skrub DataOps : les groupes de patients dans le graphe

| | |
|---|---|
| **Script** | `experiments/05_dataops.py` |
| **Hub keys** | `05_dataops` — [EstimatorReport](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/estimators/42642) (URL Kaggle) · `05_dataops_cv` — [CV 5 folds](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/cross-validations/42636) |
| **Submission** | `submission_05_dataops.csv` |
| **Modèle** | identique à l'étape 12 : `TableVectorizer` + HistGradientBoosting |

---

## Résultats

| Modèle | RMSE (GroupKFold par patient) |
|---|---|
| HistGradientBoosting, 8 numériques | 7.44 ± 0.14 |
| tabular_pipeline | 7.43 ± 0.13 |
| **DataOps (même modèle)** | **7.43 ± 0.13** |

Holdout (EstimatorReport) : 7.49. Même score que l'étape 12, comme attendu : le modèle est le même, seule l'**organisation** change.

---

## Pourquoi DataOps

Jusqu'ici, on manipulait séparément `visits`, `X`, `y` et la liste `cv_splits`. Facile de se tromper de colonnes ou d'oublier les `groups` après un copier-coller. Avec DataOps, la recette est écrite **une fois**, sous forme de graphe :

```python
data   = skrub.var("visits", visits)          # entrée nommée
groups = data["patient_id"]                    # pris AVANT de retirer les ids
X_op   = data.drop(columns=["target", "Index", "patient_id"], errors="ignore") \
             .skb.mark_as_X(cv=GroupKFold(5), split_kwargs={"groups": groups})
y_op   = data["target"].skb.mark_as_y()
pred   = X_op.skb.apply(skrub.TableVectorizer()) \
             .skb.apply(HistGradientBoostingRegressor(random_state=0), y=y_op)
learner = pred.skb.make_learner()
```

- `mark_as_X(cv=..., split_kwargs={"groups": ...})` attache le **GroupKFold par patient** directement aux features : le découpage ne peut plus être oublié.
- `errors="ignore"` sur le `drop` : `X_test` n'a pas de colonne `target`, le même graphe sert donc à prédire.
- Pour prédire, on passe un dictionnaire : `learner.predict({"visits": X_test})`.

## Deux corrections par rapport au guide

| Guide (GUIDED.md) | Ce qui marche avec skore 0.26 |
|---|---|
| `evaluate(pred)` | lève `ValueError` → `evaluate(learner, data={"visits": visits})` (le cv et les groupes sont bien lus depuis le DataOp) |
| `data.drop("target", axis=1)` | garde `Index` et `patient_id` comme features (fuite) → on les retire aussi |

## Suite

Le guide s'arrête ici. On plafonne vers 7.4 parce que chaque visite est prédite seule. Prochaine étape : `06_patient_features`.

## Commandes

```bash
python experiments/05_dataops.py --no-hub   # test local
python experiments/05_dataops.py            # push Hub (2 rapports) + submission
```
