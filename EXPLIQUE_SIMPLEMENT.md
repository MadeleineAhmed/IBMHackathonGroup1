# Ce qu'on a fait, expliqué simplement

## Le jeu qu'on joue

Les personnes atteintes de Parkinson voient leur médecin plusieurs fois par an. À chaque visite, le médecin leur donne un **score "à quel point tu trembles / tu es raide"**. Plus le score est élevé, plus c'est grave.

Mais le score du médecin est **brouillé** :

- 💊 **Le médicament cache les symptômes.** Si le patient a pris son comprimé il y a une heure, il a l'air mieux qu'il n'est vraiment. C'est comme mesurer la fatigue de quelqu'un juste après qu'il a bu un café.
- 🙅 **Le test "sans médicament" est souvent sauté**, parce que c'est pénible (le patient doit arrêter ses pilules). Résultat : 4 fois sur 10, ce score est tout simplement manquant.
- 🎲 **Tous les médecins ne notent pas pareil.**

Les organisateurs ont recalculé le **vrai score propre** pour chaque visite. Notre mission : **deviner ce vrai score en utilisant seulement les données brouillées.**

On est notés sur **l'écart moyen entre nos estimations et la réalité** (en points de score). Moins c'est grand, mieux c'est.

---

## L'idée centrale

Imagine la **taille** d'un enfant mesurée à chaque visite chez le médecin, mais avec une règle qui tremble. Chaque mesure est un peu fausse.

- Si tu regardes **une seule** mesure, tu es coincé avec son erreur.
- Si tu regardes **toutes** les mesures ensemble, tu peux tracer une belle courbe de croissance lisse, et cette courbe est bien plus proche de la vérité qu'une mesure isolée.

C'est exactement ce qui se passe ici. Le vrai score d'un patient est une **courbe lisse qui monte doucement avec les années** (la maladie s'aggrave lentement, elle ne s'améliore jamais). Chaque visite est une **mesure brouillée** de cette courbe.

Le guide officiel regardait **chaque visite séparément**. On a regardé **toutes les visites d'un patient ensemble**. C'est pour ça qu'on fait tellement mieux.

---

## Comment on a vérifié qu'on ne trichait pas

La compétition nous teste sur des **patients qu'on n'a jamais vus**.

Alors quand on s'entraînait, on **cachait des patients entiers** : on apprenait sur 80 % des patients et on testait sur les 20 % restants. Comme réviser avec un jeu de questions et être interrogé sur un autre jeu. Si on avait mélangé le même patient des deux côtés, c'est comme avoir les réponses de l'examen à l'avance : bon score à l'entraînement, mauvais en vrai.

On a fait ça **de la même façon à chaque essai**, pour pouvoir comparer les modèles entre eux honnêtement.

---

## L'escalier : ce qu'on a fait, étape par étape

Le nombre, c'est **l'écart moyen de nos prédictions sur Kaggle** (plus petit = mieux).

| Étape | Ce qu'on a fait, en mots simples | Écart moyen |
|---|---|---|
| 1 | Prédire **le même nombre pour tout le monde** (la moyenne). Idiot exprès, juste pour avoir quelque chose à battre. | 16,4 |
| 2 | Une formule simple : "plus d'années de maladie + score mesuré plus mauvais → vrai score plus mauvais". Avec une astuce : **on dit au modèle quand une mesure manque**, parce que "le médecin a sauté le test" est une information en soi. | 8,4 |
| 3 | Un modèle plus malin : plein de petits **"si ça, alors ça"** empilés (arbres de décision). Il gère les valeurs manquantes tout seul. | 7,2 |
| 4–5 | On lui a donné les colonnes **gène** et **groupe**. Ça n'a rien changé : ces colonnes ne prédisent pas vraiment la vitesse de progression. (C'est là que le guide officiel s'arrête.) | 7,1 |
| 6 | 💡 **Le grand saut.** Pour chaque visite, on a aussi dit au modèle tout ce qu'on sait des **autres visites du même patient** : son score moyen, la direction que ça prend, etc. L'astuce de la règle qui tremble. | **3,8** |
| 7 | **Annuler l'effet du médicament.** On a appris "1 heure après le comprimé, le score semble deux fois moins grave qu'il l'est vraiment ; 3 heures après, un peu moins masqué…" et on a corrigé chaque mesure avant de l'utiliser. | 3,7 |
| 8 | **Relier les points en douceur.** Le vrai score est une courbe lisse, donc on a lissé les prédictions de chaque patient en une courbe douce plutôt qu'en zigzags. | 3,6 |
| 9 | On a laissé la courbe du patient **se courber légèrement** (la progression réelle n'est pas une droite parfaite), et on a dit au modèle **à quel point les mesures de chaque patient sont fiables**. | 3,5 |
| 10 | **On a tourné les boutons** du modèle (vitesse d'apprentissage, prudence…) en testant 30 combinaisons et en gardant la meilleure. | 3,4 |
| 11 | **On a demandé à 4 modèles et on a fait la moyenne de leurs réponses**, comme demander à plusieurs amis et prendre la réponse du milieu. | 3,35 |
| 12 | 💡 **Chacun réagit différemment au médicament.** Pour certains le comprimé marche très bien, pour d'autres moins. Quand un patient a fait les deux tests (avec et sans médicament) à certaines visites, on a mesuré **sa propre réaction** et on l'a utilisée pour corriger ses autres visites. | **3,14** |

**De 16,4 → 3,14.** Le meilleur score du guide officiel était 7,1 — on se trompe de **moins de moitié** autant.

---

## Pourquoi on a choisi le modèle de l'étape 12

- 🏆 **C'est le meilleur** sur nos tests d'entraînement *et* sur le vrai test de Kaggle.
- ✅ **Nos scores d'entraînement ont toujours correspondu à ceux de Kaggle**, donc on fait confiance au résultat, ce n'est pas de la chance.
- 🧠 **Chaque élément a une raison** qui vient des données ou du fonctionnement de la maladie et du médicament : courbe de maladie lisse, médicament qui s'estompe en quelques heures, réaction personnelle au traitement.
- 🧼 **C'est un seul modèle**, pas une pile de modèles, donc plus simple à expliquer — et il bat quand même la pile de l'étape 11.

---

## Les choses qu'on a essayées et qui n'ont pas marché (et c'est normal)

Montrer ce qui n'a *pas* marché prouve qu'on a bien tout testé.

- Changer le bouton "alpha" de la formule simple → aucun effet.
- Utiliser le gène et le groupe → aucun effet.
- Lisser avec une **droite** plutôt qu'une courbe → moins bien (la vraie courbe se coude légèrement).
- Une méthode statistique avancée pour deviner la courbe de chaque patient → pas mieux que ce qu'on avait.
- Corriger le score "sans médicament" en *pourcentage* plutôt qu'en *"moins quelques points"* → moins bien.

---

## Comment on a travaillé (la "méthodologie" en une respiration)

1. **Regarder les données d'abord** et les comprendre (c'est comme ça qu'on a trouvé l'idée de la courbe lisse).
2. **Changer une seule chose à la fois.**
3. **Tester honnêtement** (cacher des patients entiers).
4. **Garder uniquement si ça améliore les résultats.**
5. **Regarder où on se trompe encore**, comprendre pourquoi, et corriger ça ensuite.
6. **Tout écrire** : chaque essai a son script, son explication, son rapport sur Skore Hub et sa soumission Kaggle.

---

## Si quelqu'un vous pose des questions difficiles

**"Est-ce que regarder les autres visites du même patient, ce n'est pas de la triche ?"**
Non. On utilise seulement les mesures brouillées, jamais les réponses, et Kaggle nous donne le même type d'historique de visites pour les patients du test aussi.

**"Comment savez-vous que ce n'est pas juste de la chance ?"**
On a toujours testé sur des patients que le modèle n'avait jamais vus, et les scores Kaggle (patients totalement inconnus) ont correspondu aux nôtres à chaque fois.

**"Quelle a été la chose la plus importante que vous avez faite ?"**
Regarder les données avant de construire des modèles. C'est comme ça qu'on a remarqué que chaque patient suit une courbe lisse — et utiliser ça a divisé l'erreur par deux.
