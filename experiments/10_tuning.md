# 10 — Réglage des hyperparamètres (et choix des colonnes)

| | |
|---|---|
| **Scripts** | `experiments/10_tuning.py` (modèle final) · `experiments/10_tuning_search.py` (la recherche, ~16 min) |
| **Hub keys** | `10_tuning` — [EstimatorReport](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/estimators/43284) (URL Kaggle) · `10_tuning_cv` — [CV 5 folds](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/cross-validations/43278) |
| **Submission** | `submission_10_tuning.csv` |
| **Modèle** | identique à `09` (mêmes features, même lissage) ; seuls les réglages du HistGradientBoosting changent |

---

## Résultats

| Modèle | RMSE (GroupKFold par patient) |
|---|---|
| 09 — tendance courbe | 3.50 ± 0.08 |
| **10 — réglages optimisés** | **3.47 ± 0.08** |

Holdout (EstimatorReport) : 3.58. Gain modeste (−0.03) : **ce ne sont plus les réglages qui limitent le modèle**, ce sont les informations disponibles.

---

## D'abord : « alpha » n'est pas la combinaison de colonnes

C'est une confusion fréquente, d'où cette section.

- **La combinaison de colonnes**, ce sont les **poids** que le modèle apprend (pour le Ridge : un coefficient par colonne, cf. le tableau de `02_ridge.md`). On ne les choisit pas à la main : l'entraînement les calcule.
- **`alpha`** (Ridge) est un **hyperparamètre** : un réglage *choisi avant* l'entraînement, qui dit à quel point tirer ces poids vers zéro (un bouton de « prudence »). Sur nos données, `alpha` de 0.1 à 100 donnait le même score (8.53).

Le vrai objectif, « trouver la configuration qui donne le meilleur score », se traduit en deux questions :

1. **Quels réglages (hyperparamètres)** pour le modèle ? → recherche ci-dessous ;
2. **Quelles colonnes** lui donner ? → test de sélection ci-dessous.

## 1. Recherche des hyperparamètres

Le HistGradientBoosting a ses propres « alpha ». Chacun a été testé, **toujours sur les mêmes 5 folds groupés par patient** :

| Réglage | Ce qu'il contrôle | Valeurs testées |
|---|---|---|
| `learning_rate` | taille de chaque correction apportée par un nouvel arbre | 0.02, 0.03, 0.05, 0.1 |
| `max_iter` | nombre d'arbres | 300, 500, 800, 1200 |
| `max_leaf_nodes` | complexité de chaque arbre (nombre de feuilles) | 15, 31, 63 |
| `min_samples_leaf` | nombre minimum de visites par feuille (plus grand = plus prudent) | 20, 50, 100, 200, 400 |
| `max_features` | part des colonnes vue à chaque coupure (ajoute de la diversité) | 50 %, 80 %, 100 % |
| `l2_regularization` | pénalité sur les valeurs des feuilles (l'équivalent le plus direct d'`alpha`) | 0, 0.1, 1, 10 |

4 × 4 × 3 × 5 × 3 × 4 = 2 880 combinaisons possibles ; une **recherche aléatoire** en teste 30 (`RandomizedSearchCV`), ce qui suffit en pratique à trouver une zone proche de l'optimum.

### Meilleures combinaisons

Scores sans le lissage (`RandomizedSearchCV`), puis confirmés avec le lissage par patient :

| # | learning_rate | arbres | feuilles | min visites / feuille | colonnes / coupure | L2 | RMSE | RMSE + lissage |
|---|---|---|---|---|---|---|---|---|
| **1** | **0.02** | **1200** | **63** | **200** | **80 %** | **0** | **3.506** | **3.467** |
| 2 | 0.03 | 1200 | 31 | 50 | 100 % | 1 | 3.511 | 3.468 |
| 3 | 0.05 | 500 | 63 | 20 | 50 % | 0 | 3.513 | 3.469 |
| réglages de `09` (approx.) | 0.05 | 500 | 31 | 100 | 100 % | 0 | ~3.54 | 3.497 |

**Lecture :** les meilleures configurations **apprennent lentement avec beaucoup d'arbres** (petit `learning_rate`, `max_iter` élevé). La pénalité L2 compte peu, comme `alpha` pour le Ridge. Les trois premières sont à 0.002 près : le score est **stable**, on n'est pas tombé sur un réglage chanceux.

**Précaution :** les réglages sont choisis sur les mêmes folds qui servent à les évaluer, donc 3.47 est très légèrement optimiste. Le score Kaggle (données jamais vues) sert de contrôle.

## 2. Sélection des colonnes

skore signalait `sexM`, `gene` et `cohort` comme inutiles (check SKD012). Testé avec les réglages de `09` :

| Colonnes | RMSE |
|---|---|
| toutes (39) | 3.497 |
| sans `sexM`, `gene`, `cohort` | 3.498 |
| + sans les heures de prise brutes | 3.506 |
| sans les tendances linéaires (on garde les paraboles) | 3.498 |

Retirer ces colonnes **ne change rien** : le modèle les ignore déjà. Les heures de prise brutes, elles, apportent encore un peu. **On garde toutes les colonnes.**

## Commandes

```bash
python experiments/10_tuning_search.py        # refaire la recherche (~16 min)
python experiments/10_tuning.py --no-hub      # test local du modèle retenu
python experiments/10_tuning.py               # push Hub (2 rapports) + submission
```
