# Étapes 10–11 — CV par patient + HistGradientBoosting

| | |
|---|---|
| **Script** | `experiments/03_hgbr.py` |
| **Hub keys** | `03_hgbr` — [EstimatorReport](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/estimators/42609) (URL Kaggle) · `03_hgbr_cv` — [CV 5 folds](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/cross-validations/42602) |
| **Submission** | `submission_03_hgbr.csv` |
| **Features** | les 8 colonnes numériques, **sans imputation** |

---

## Résultats

| Modèle | Évaluation | RMSE |
|---|---|---|
| Dummy | split aléatoire | 16.48 |
| Ridge + indicateurs | GroupKFold par patient | 8.54 |
| **HistGradientBoosting** | **GroupKFold par patient (5 folds)** | **7.44 ± 0.14** |
| HistGradientBoosting | holdout ~20 % des patients (EstimatorReport) | 7.48 |

−13 % d'erreur par rapport au Ridge, avec les mêmes 8 colonnes.

---

## Étape 10 — Évaluer par patient (GroupKFold)

Chaque patient a 4 à 12 visites, et les patients de `X_test` n'apparaissent jamais dans le train. Un split aléatoire par visite met des visites du même patient des deux côtés : le modèle « reconnaît » la personne et le score local est trop optimiste.

`GroupKFold(n_splits=5)` sur `patient_id` découpe les **patients** en 5 groupes ; on entraîne sur 4 et on évalue sur le 5e, cinq fois. skore appelle `splitter.split(X, y)` sans `groups=`, donc les paires d'indices sont précalculées (`parkinson.data.grouped_cv_splits`).

À partir d'ici, **tous les modèles sont comparés sur ce même découpage**.

## Étape 11 — HistGradientBoosting

- Des centaines de petits arbres de décision, chacun corrigeant les erreurs des précédents (boosting). Capte les effets non linéaires et les interactions (ex. le biais du score OFF qui dépend de l'heure de la dernière dose).
- **Valeurs manquantes gérées nativement** : à chaque coupure, l'arbre apprend de quel côté envoyer les `NaN`. « OFF non mesuré » devient une information, sans imputation.
- Hyperparamètres par défaut, `random_state=0`.

## Deux rapports sur le Hub

Le règlement demande une URL d'**EstimatorReport** dans la description Kaggle. On pousse donc :

- `03_hgbr` : `EstimatorReport` entraîné sur ~80 % des patients, évalué sur les ~20 % restants (premier fold du GroupKFold) → **URL collée sur Kaggle** ;
- `03_hgbr_cv` : `CrossValidationReport` sur les 5 folds → le score le plus fiable pour comparer les modèles.

La submission, elle, vient du modèle réentraîné sur **toutes** les visites du train.

---

## Limites

| Limite | Étape qui la traite |
|---|---|
| `gene` et `cohort` (texte) ignorés | Étape 12 — skrub `tabular_pipeline` |
| Chaque visite prédite seule, sans les autres visites du même patient | Au-delà du guide — features par patient (`06_patient_features`) |

## Commandes

```bash
python experiments/03_hgbr.py --no-hub   # test local
python experiments/03_hgbr.py            # push Hub (2 rapports) + submission
```
