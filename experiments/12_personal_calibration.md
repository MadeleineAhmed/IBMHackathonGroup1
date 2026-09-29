# 12 — Réponse personnelle au médicament + motifs de valeurs manquantes

| | |
|---|---|
| **Script** | `experiments/12_personal_calibration.py` |
| **Code réutilisable** | `src/parkinson/features.py` — `PatientPersonalFeatures` |
| **Hub keys** | `12_personal_calibration` — [EstimatorReport](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/estimators/43721) (URL Kaggle) · `12_personal_calibration_cv` — [CV 5 folds](https://skore.probabl.ai/ibmhackathongroup1/ibm-hackathon/cross-validations/43715) |
| **Submission** | `submission_12_personal_calibration.csv` |
| **Modèle** | `PatientSmoother(degree=2)` autour de `PatientPersonalFeatures` → HistGradientBoosting (réglages de `10`) |

---

## Résultats

| Modèle | RMSE (GroupKFold par patient) |
|---|---|
| 10 — meilleur modèle seul | 3.47 ± 0.08 |
| + motifs de valeurs manquantes seuls | 3.45 |
| + calibration personnelle ON seule | 3.32 |
| **12 — les deux** | **3.29 ± 0.06** |

Holdout (EstimatorReport) : 3.37. Plus gros gain depuis `09`.

---

## Pourquoi : l'analyse d'erreur de `10`

| Constat sur les prédictions hors-fold de `10` | Valeur |
|---|---|
| Part de l'erreur qui est un décalage de niveau par patient | 65 % |
| RMSE des visites **ON seulement** (examen OFF sauté) | 3.9–4.2 |
| RMSE des visites avec une mesure OFF (sans heure de prise) | 2.4 |
| Part de la variation du ratio ON / vrai score qui est **propre au patient** | 39 % |

La correction de `07` utilise **une seule courbe pour toute la population** : « à 2 h après la prise, le score ON vaut ~45 % du vrai score ». Mais certains patients répondent plus fortement à la lévodopa que d'autres. Pour un patient qui répond fort, la courbe moyenne sous-estime son vrai score ; pour ses visites où l'examen OFF a été sauté, rien ne rattrape l'erreur.

## Comment ça marche

### 1. Facteur personnel de réponse au médicament

Pour chaque patient, les visites où **ON et OFF ont tous deux été mesurés** permettent de comparer ses deux estimations corrigées :

```
facteur_personnel = médiane( est_on / est_off )   sur ses visites avec les deux mesures
```

- Si le patient répond comme la moyenne, le facteur vaut ~1 ; s'il répond plus fort (ON plus bas que prévu), il est < 1.
- Le facteur est **rétréci vers 1** selon le nombre de visites disponibles : `facteur ^ (n / (n + 2))`. Avec une seule visite, on ne lui fait qu'à moitié confiance.
- Chaque mesure ON du patient est ensuite divisée par ce facteur, puis recombinée avec les mesures OFF comme en `07`, et on recalcule sa tendance (droite et parabole).

Sur les visites ON seulement (folds de validation), l'erreur de l'estimation par visite passe de **13.9 à 13.1** ; 91 % de ces visites appartiennent à un patient qui a au moins une visite avec les deux mesures.

Aucune fuite : le facteur n'utilise que les colonnes d'entrée du patient (les courbes de population restent apprises dans chaque fold d'entraînement).

### 2. Motifs de valeurs manquantes (B10)

Quelles mesures manquent **ensemble** porte une information sur le protocole de la visite (par ex. visite « ON seulement »). On ajoute :

- `miss_code` : un code par visite indiquant lesquelles de `ledd`, `on`, `off` et des deux heures de prise manquent ;
- `p_miss_*` : pour chaque colonne, la part des visites du patient où elle manque.

Gain seul modeste (3.47 → 3.45), un peu complémentaire de la calibration personnelle.

## Idées testées et écartées au passage

| Idée | Résultat |
|---|---|
| Correction OFF **multiplicative** (ratio, comme pour ON) au lieu d'additive | pire : erreur par visite 9.18 contre 7.85 |
| Correction OFF selon l'heure **et** la durée de maladie | pas mieux par visite (7.90) : le biais OFF augmente avec la durée, mais le bruit domine |
| Courbe bayésienne empirique par patient (B9) | 3.472 contre 3.467 : pas de gain (journal : abandonné) |

## Commandes

```bash
python experiments/12_personal_calibration.py --no-hub   # test local
python experiments/12_personal_calibration.py            # push Hub (2 rapports) + submission
```
