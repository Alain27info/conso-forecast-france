# conso-forecast

Prévision de la consommation électrique française à J+1, avec comparaison de plusieurs approches (baselines, arbres boostés, SARIMA, réseau récurrent) et mise à disposition du meilleur modèle via une API conteneurisée.

> **Statut : en cours de développement.** Les sections ci-dessous ne contiennent que des résultats réellement obtenus et reproductibles. Le réseau récurrent, l'API et Docker ne sont pas encore faits.

## Problème

Prévoir la consommation nationale d'électricité pour chaque pas de 30 minutes du lendemain. Sur un réseau électrique, la production doit égaler la consommation à chaque instant : une bonne prévision permet de préparer la production à l'avance.

## Données

- **Source** : RTE, *Données éCO2mix nationales consolidées et définitives* (open data), [Licence Ouverte v2.0 (Etalab)](https://www.etalab.gouv.fr/wp-content/uploads/2017/04/ETALAB-Licence-Ouverte-v2.0.pdf).
- **Période** : du 1er janvier 2012 au 31 août 2026 (heure de Paris), 257 134 créneaux de 30 minutes. Les données sont *définitives* jusqu'au 31 décembre 2025, *consolidées* ensuite.
- **Traitement** : index en UTC ; 60 horodatages dupliqués au changement d'heure du printemps (moyennés) ; une heure manquante chaque automne (laissée vide, pas interpolée, pour éviter d'utiliser une valeur du futur). Voir `src/data.py` et `notebooks/01_exploration.ipynb`.
- Les données brutes ne sont pas versionnées : `python -m src.data` les télécharge.

## Méthode

- **Test** : les 12 derniers mois (du 1er septembre 2025 au 31 août 2026), puis validation en origine glissante sur 5 années de test successives (voir Résultats). Pas de découpage aléatoire.
- **Règle d'information** : la prévision du jour D n'utilise que les données jusqu'à la veille 23:30. Le code coupe l'historique avant d'appeler le modèle (`src/evaluate.py`), et un test vérifie qu'en remplaçant tout le futur par du bruit, la prévision ne change pas.
- **Variables** (`src/features.py`) : calendrier (heure, jour de la semaine, mois, jours fériés calculés sans bibliothèque) et retards de consommation d'**au moins un jour** (1, 2, 7, 14 jours, moyennes de la veille et des 7 jours précédents). Aucun retard plus court : il ne serait pas connu à J+1. Des tests vérifient que les variables du jour D ne changent pas si l'on remplace le futur par du bruit.
- **Évaluation générique** : mêmes lignes, mêmes métriques (MAE, RMSE, MAPE) pour tous les modèles.

## Résultats

Baselines (`python -m src.run_baselines`) et gradient boosting (`python -m src.run_gbm`). Scores sur 17 507 créneaux de la période de test (1er septembre 2025 – 31 août 2026) :

| Modèle | MAE (MW) | RMSE (MW) | MAPE (%) |
|---|---|---|---|
| Baseline : même heure la veille | 2 444 | 3 574 | 4,86 |
| Baseline : même heure la semaine précédente | 3 445 | 5 066 | 6,42 |
| Gradient boosting, calendrier seul | 3 953 | 5 157 | 7,56 |
| **Gradient boosting, calendrier + retards** | **1 318** | 1 873 | **2,46** |
| Prévision J-1 de RTE (référence externe) | 1 388 | 1 747 | 2,74 |

- Le gradient boosting réduit la MAE de 46 % par rapport à la baseline « veille ». Le gain vient surtout du lundi (MAE 5 306 → 1 668 MW) et du samedi (4 190 → 1 333 MW) ; le mercredi et le jeudi, la baseline « veille » est légèrement meilleure que le modèle.
- Sans les retards de consommation, le modèle est moins bon que la baseline la plus simple : l'information utile est dans la consommation récente.
- Le nombre d'arbres (517) et la fonction de perte sont choisis sur une période de validation de 12 mois précédant le test ; le test ne sert à aucun réglage.
- **La prévision de RTE n'est pas directement comparable** : elle est réalisée plus tôt, avec des informations que ce projet n'utilise pas forcément. Elle sert seulement de point de repère, et l'écart de MAE ne prouve pas que le modèle fait mieux.

### Validation en origine glissante

`python -m src.run_rolling` (~12 min). Cinq plis : chaque pli teste une année (1er septembre – 31 août) en s'entraînant sur tout le passé avant cette année ; la perte et le nombre d'arbres sont réglés sur les 12 mois précédant le test du pli. MAE en MW :

| Pli (année de test) | Baseline veille | Gradient boosting | GB, écart à la veille | RTE J-1 (référence) |
|---|---|---|---|---|
| 2021-22 | 2 737 | 1 211 | 1 162 | 967 |
| 2022-23 | 2 575 | 1 209 | 1 145 | 1 191 |
| 2023-24 | 2 514 | 1 292 | 1 186 | 1 369 |
| 2024-25 | 2 536 | 1 343 | 1 243 | 1 310 |
| 2025-26 | 2 444 | 1 318 | 1 288 | 1 388 |
| **Moyenne** | 2 561 | 1 275 | **1 205** | 1 245 |
| Écart-type | 109 | 62 | 60 | 173 |

- Le gradient boosting bat la baseline « veille » sur les 5 plis (MAE −46 % à −56 % selon le pli, −50 % en moyenne). Le gain diminue d'un pli à l'autre (−56 % puis −46 %) : la MAE de la baseline baisse au fil des années alors que celle du modèle tend à monter ; la cause n'est pas étudiée. Le léger désavantage du mercredi et du jeudi observé sur le dernier pli ne se retrouve pas sur les 4 autres plis.
- « GB, écart à la veille » prédit y − (consommation de la veille) au lieu de y. Sa MAE est plus basse sur les 5 plis (−5,5 % en moyenne par rapport au gradient boosting classique) ; il a aussi la MAE de validation la plus basse sur les 5 plis, donc il aurait été choisi sans regarder les tests. L'idée est venue après avoir regardé le dernier pli seulement.
- **Face à RTE, le résultat dépend du pli** : RTE est meilleur en MAE sur 2021-22 et, pour le gradient boosting classique, aussi sur 2022-23 et 2024-25 ; il est meilleur en RMSE sur les 5 plis. Le seul test de l'an dernier (1 318 contre 1 388) était donc la fenêtre la plus favorable ; la comparaison reste non équivalente.
- Le modèle surestime en moyenne sur chaque pli (+60 à +266 MW pour le gradient boosting classique, la baseline « veille » est à ±20 MW). Pour le gradient boosting classique, les erreurs sont plus fortes l'hiver (MAE 2 092 MW en hiver contre 729 MW en été) et en journée (1 300 à 1 460 MW entre 7h et 18h, contre environ 1 040 MW la nuit) ; les pires journées (erreur de 6 000 à 9 000 MW) sont surtout en hiver, avec des surestimations comme des sous-estimations. Leurs causes n'ont pas été vérifiées.

### SARIMA

`python -m src.run_sarima` (~13 min, les plis tournent en parallèle ; nécessite d'avoir lancé `run_rolling` avant). Même protocole que ci-dessus (mêmes plis, mêmes règles d'information). Modèle : différence saisonnière à 24 h (48 pas de 30 minutes), ARMA(p, q) et MA saisonnier sur la série différenciée, avec jour de la semaine et jours fériés comme variables externes. Les paramètres sont estimés une seule fois par pli sur les 56 jours précédant le test, puis le filtre est mis à jour chaque jour sans ré-estimation ; trois spécifications sont comparées sur les 28 jours précédant chaque pli. MAE en MW, sur les mêmes lignes pour tous les modèles (11 créneaux sur 87 648 écartés faute de prévision SARIMA, d'où des écarts de 1 MW avec le tableau précédent) :

| Pli | Baseline veille | Gradient boosting | SARIMA |
|---|---|---|---|
| 2021-22 | 2 737 | 1 212 | 2 555 |
| 2022-23 | 2 575 | 1 209 | 3 193 |
| 2023-24 | 2 514 | 1 293 | 2 212 |
| 2024-25 | 2 536 | 1 344 | 2 265 |
| 2025-26 | 2 444 | 1 319 | 2 411 |
| **Moyenne** | 2 561 | 1 275 | 2 527 |
| Écart-type | 109 | 62 | 395 |

- **SARIMA fait à peu près comme la baseline « veille »** (MAE moyenne −1,3 %, meilleur sur 4 plis sur 5, nettement moins bon sur 2022-23) et son erreur est environ **le double** de celle du gradient boosting. Il est moins bon que RTE sur les 5 plis.
- Par jour de la semaine, il n'améliore la baseline « veille » que le lundi (MAE 3 387 contre 5 545 MW) et le samedi (3 534 contre 4 249) ; du mardi au dimanche il est moins bon qu'elle. Il est meilleur que le gradient boosting seulement entre minuit et 2h (par exemple 795 contre 1 038 MW à minuit), c'est-à-dire juste après la dernière observation.
- Ce résultat est celui d'**une version simplifiée** : une seule saisonnalité (le jour) au lieu de deux, paramètres estimés sur 56 jours et jamais ré-estimés, trois spécifications seulement comparées, aucune identification des ordres par autocorrélation. Il ne permet pas de conclure que SARIMA est inadapté en général.

## Limites

- Pas de météo : la température, qui influence fortement la consommation (chauffage), n'est pas utilisée. Les erreurs sont plus fortes l'hiver, ce qui est compatible avec un effet du chauffage, mais ce lien n'est pas démontré ici.
- Le nombre d'arbres est plafonné à 600 lors du réglage, et ce plafond est atteint (ou presque) pour le gradient boosting classique sur 3 des 5 plis. Contrôle sur un seul pli (2021-22, MAE de validation uniquement) : avec un plafond de 1 500 arbres, la MAE de validation passe de 1 316 à 1 306 MW. L'effet paraît faible mais n'a pas été mesuré sur les autres plis.
- Les événements exceptionnels (canicule, crise énergétique, COVID) ne sont pas modélisés.
- Les causes exactes de certaines anomalies des données (doublons, heure manquante) ne sont pas documentées par la source.

## Installation et lancement

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
pytest
python -m src.data               # télécharge les données (~33 Mo dans data/)
python -m src.run_baselines      # affiche le tableau des baselines
python -m src.run_gbm            # entraîne le gradient boosting et le compare (~2 min)
python -m src.run_rolling        # validation en origine glissante, 5 plis (~12 min)
python -m src.run_sarima         # SARIMA sur les mêmes plis (~13 min), après run_rolling
```

Pour ouvrir le notebook dans Jupyter, enregistrer d'abord le noyau du projet :

```bash
python -m ipykernel install --user --name conso-forecast --display-name "conso-forecast (.venv)"
```

Testé avec Python 3.12 sur Windows 11.

## Structure du dépôt

```
conso-forecast/
├── data/            (non versionné : voir src/data.py pour le téléchargement)
├── notebooks/       (exploration)
├── src/             (données, variables, métriques, baselines, évaluation)
├── api/             (FastAPI, à venir)
├── tests/
├── models/          (modèles sauvegardés, non versionnés)
└── requirements.txt
```
