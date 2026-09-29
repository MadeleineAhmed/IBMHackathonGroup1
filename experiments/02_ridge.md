# Étape 8 — Ridge : premier modèle linéaire

## Ce qu'on a fait

On a remplacé le **dummy** (qui prédisait toujours la même valeur) par un **modèle linéaire Ridge**.
Ridge regarde les features numériques du patient (âge, dose, scores cliniques) et apprend
une combinaison de poids pour prédire le vrai score OFF.

Script : `experiments/02_ridge.py`  
Hub key : `02_ridge`  
Submission : `submission_02_ridge.csv`

---

## Résultats

| Modèle | RMSE | Gain vs dummy |
|---|---|---|
| Dummy (moyenne = 37.47) | 16.48 | référence |
| **Ridge (alpha=1.0)** | **10.39** | **−37%** |

Le Ridge bat largement le dummy. Les features numériques portent un vrai signal.

### Comparaison des alphas testés

| alpha | RMSE Ridge |
|---|---|
| 0.1 | 10.3869 |
| **1.0** | **10.3869** |
| 10.0 | 10.3869 |
| 100.0 | 10.3870 |

→ L'alpha a très peu d'impact ici. On garde `alpha=1.0` (valeur par défaut).

---

## Comment ça marche — Ridge expliqué

### Le dummy (étape précédente)

```python
DummyRegressor(strategy="mean")
# Prédit toujours : 37.47
# Ignore toutes les features
```

```
Visite patient A (age=52, off=44)  →  37.47
Visite patient B (age=70, off=NaN) →  37.47   ← même chose pour tout le monde
```

### Ridge

```python
make_pipeline(
    SimpleImputer(strategy="median"),
    Ridge(alpha=1.0),
)
```

Ridge apprend une équation linéaire :

```
target ≈ intercept
       + poids_age              × age
       + poids_age_at_diagnosis × age_at_diagnosis
       + poids_ledd             × ledd
       + poids_on               × on
       + poids_off              × off
       + poids_time_on          × time_since_intake_on
       + poids_time_off         × time_since_intake_off
       + poids_sexM             × sexM
```

**Exemple concret :**
```
Patient 52 ans, off=44, ledd=607  →  Ridge prédit ≈ 38.5
Patient 70 ans, off=80, ledd=900  →  Ridge prédit ≈ 55.2
Patient 48 ans, off=NaN, ledd=NaN →  Ridge prédit ≈ 31.0  (NaN remplacés par médiane)
```

---

## Le rôle de SimpleImputer

Ridge **ne peut pas** travailler avec des valeurs manquantes (`NaN`).
On doit donc remplir les trous **avant** de lui donner les données.

```
Données brutes          Après SimpleImputer       Ridge prédit
off = NaN        →      off = 37.0 (médiane)  →   target = 34.2
ledd = NaN       →      ledd = 600.0 (médiane) →
```

### Pourquoi c'est une limitation

En remplissant les NaN avec la médiane, on **cache le signal** :

```
off = NaN  →  signifie : "l'exam OFF était trop inconfortable, il a été sauté"
           →  c'est une information clinique sur la sévérité du patient
           →  SimpleImputer efface cette information
```

C'est pour ça qu'on passera à **HGBR** (étape 10) qui gère les NaN nativement
et peut apprendre que "off manquant" est un signal en soi.

---

## Le paramètre alpha — régularisation

`alpha` contrôle à quel point Ridge "bride" ses poids :

```
alpha petit (0.1)  →  poids libres  →  peut sur-apprendre le bruit
alpha grand (100)  →  poids bridés  →  modèle plus conservateur
```

**Dans nos données :** l'alpha change presque rien (RMSE identique à 4 décimales).
Cela suggère que le signal est fort et stable — le modèle n'a pas besoin de régularisation
agressive sur ces features.

---

## Ce que Ridge ne peut pas faire

| Limitation | Conséquence |
|---|---|
| Modèle **linéaire** uniquement | Ne capte pas les interactions (ex: `age × ledd`) |
| Pas de NaN natifs | Perd le signal "off manquant" |
| Pas de features catégorielles | `gene` et `cohort` ignorés |
| Pas de CV par patient | Splitter aléatoire → potentiel leakage |

Ces limites sont exactement ce qu'on corrige dans les étapes suivantes :
- **Étape 10** : GroupKFold (CV par patient, pas de leakage)
- **Étape 11** : HGBR (NaN natifs, interactions)
- **Étape 12** : skrub tabular_pipeline (ajoute `gene` / `cohort`)

---

## Commandes utilisées

```powershell
# Test local (sans push Hub)
python experiments/02_ridge.py --no-hub

# Test avec différents alpha
python experiments/02_ridge.py --no-hub --alpha 0.1
python experiments/02_ridge.py --no-hub --alpha 10.0
python experiments/02_ridge.py --no-hub --alpha 100.0

# Push Hub + génération submission
python experiments/02_ridge.py
```

## Fichiers produits

| Fichier | Contenu |
|---|---|
| `submission_02_ridge.csv` | Prédictions sur X_test à uploader sur Kaggle |
| Hub key `02_ridge` | Rapport skore avec comparaison dummy vs ridge |
