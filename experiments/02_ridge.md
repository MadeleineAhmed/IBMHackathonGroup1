# Étape 8 — Ridge : premier modèle linéaire

| | |
|---|---|
| **Script** | `experiments/02_ridge.py` |
| **Hub key** | `02_ridge` — [rapport](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/estimators/42552) |
| **Submission** | `submission_02_ridge.csv` |
| **Features** | 8 colonnes numériques : `sexM`, `age_at_diagnosis`, `age`, `ledd`, `time_since_intake_on`, `time_since_intake_off`, `on`, `off` |

---

## Résultats

| Modèle | RMSE (split aléatoire 80/20) | RMSE (GroupKFold par patient) | Gain vs dummy |
|---|---|---|---|
| Dummy (moyenne = 37.47) | 16.48 | — | référence |
| Ridge, imputation médiane seule | 10.39 | 10.44 | −37 % |
| **Ridge + indicateurs de valeurs manquantes** | **8.53** | **8.54** | **−48 %** |

- Le split aléatoire (celui du guide pour cette étape) et le GroupKFold par patient donnent quasiment le même score : un modèle linéaire ne peut pas « mémoriser » un patient, donc la fuite entre visites d'un même patient reste faible ici.
- À partir de l'étape 10, on compare tous les modèles en **GroupKFold par patient**.

### Paramètre `alpha`

| alpha | RMSE Ridge |
|---|---|
| 0.1 | 8.5305 |
| **1.0** | **8.5305** |
| 10.0 | 8.5305 |
| 100.0 | 8.5311 |

`alpha` n'a presque aucun effet : avec ~35 000 visites d'entraînement et seulement 14 colonnes, la régularisation est négligeable devant la quantité de données. On garde `alpha=1.0`.

---

## Comment ça marche

Ridge apprend une **équation linéaire** : une somme pondérée des features.

```
target ≈ intercept + Σ (poids × feature)
```

Ridge ne sait pas traiter les valeurs manquantes (`NaN`), d'où le pipeline :

```python
make_pipeline(
    SimpleImputer(strategy="median", add_indicator=True),
    Ridge(alpha=1.0),
)
```

1. **`SimpleImputer(strategy="median")`** remplace chaque `NaN` par la médiane de la colonne.
2. **`add_indicator=True`** ajoute, pour chaque colonne qui a des trous, une colonne 0/1 « cette valeur était manquante ».
3. **`Ridge`** apprend un poids pour les 8 features + les 6 indicateurs.

### Pourquoi les indicateurs changent tout (10.39 → 8.53)

Sans indicateur, un `off` manquant devient simplement `off = médiane` : le modèle ne peut plus distinguer « OFF mesuré à la valeur médiane » de « examen OFF sauté ». Or un examen sauté est une information (le patient n'a été vu qu'en ON, protocole de la cohorte…).

Avec l'indicateur, le modèle apprend une **correction** quand la valeur manquait. Poids appris (modèle entraîné sur tout le train) :

| Colonne | Poids |
|---|---|
| `on` | +0.56 |
| `off` | +0.55 |
| `age` | +0.34 |
| `age_at_diagnosis` | −0.28 |
| `time_since_intake_on` | +1.10 |
| **`on` manquant** | **−9.72** |
| **`off` manquant** | **+6.93** |
| autres indicateurs | entre −0.23 et +0.44 |

Lecture : l'indicateur corrige l'erreur introduite par le remplissage à la médiane — environ +7 points quand `off` manque, −10 points quand `on` manque. Ces deux indicateurs portent l'essentiel du gain.

### Exemples réels (deux visites du patient `IPLP5212`)

| Visite | `on` | `off` | Vrai `target` | Prédiction Ridge |
|---|---|---|---|---|
| âge 52.1 | 7 | manquant | 34.7 | 34.1 |
| âge 53.0 | 12 | 44 | 38.1 | 41.6 |

---

## Limites (corrigées dans les étapes suivantes)

| Limite | Étape qui la traite |
|---|---|
| Modèle **linéaire** : pas d'interactions (ex. effet de l'heure de la dose qui dépend du score) | Étape 11 — HistGradientBoosting |
| Split aléatoire par visite pour l'évaluation | Étape 10 — GroupKFold par patient |
| `gene` et `cohort` (texte) ignorés | Étape 12 — skrub `tabular_pipeline` |
| Chaque visite est prédite seule, sans les autres visites du même patient | Au-delà du guide — features par patient (voir `context/ml_decisions.md`) |

---

## Points techniques

- **Rapport Hub** : le script évalue dummy et Ridge ensemble (`ComparisonReport`) pour afficher la comparaison, mais `project.put()` n'accepte que `EstimatorReport` / `CrossValidationReport`. On pousse donc `report.reports_["ridge"]`. Le Hub calcule de lui-même la comparaison au dummy (check SKD002).
- **Submission** : l'`Index` est pris dans `X_test` (et vérifié contre `sample_submission.csv`) pour que chaque prédiction reste alignée sur sa visite.

---

## Commandes

```bash
# Test local, sans push Hub
python experiments/02_ridge.py --no-hub

# Autre valeur d'alpha
python experiments/02_ridge.py --no-hub --alpha 10

# Push Hub (rapport Ridge) + génération de la submission
python experiments/02_ridge.py
```

Ensuite : uploader `submission_02_ridge.csv` sur Kaggle et coller l'URL du rapport Hub (affichée par le script) dans la **Submission Description**.
