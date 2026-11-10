# conso-forecast

Prévision de la consommation électrique française à J+1, avec comparaison de plusieurs approches (baselines, arbres boostés, SARIMA, réseau récurrent) et mise à disposition du meilleur modèle via une API conteneurisée.

> **Statut : en cours de développement.** Les sections ci-dessous ne contiennent que des résultats réellement obtenus et reproductibles. Docker n'est pas encore fait.

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
- **Variables** (`src/features.py`) : calendrier (heure, jour de la semaine, mois, jours fériés calculés sans bibliothèque) et retards de consommation d'**au moins un jour** (1, 2, 7, 14 jours, moyennes de la veille et des 7 jours précédents). Pas de retard de moins d'un jour à la même heure (il ne serait pas connu pour toutes les heures du lendemain). Ces variables n'utilisent donc pas les dernières heures de la veille, qui sont pourtant connues : voir la section « Réseau récurrent ». Des tests vérifient que les variables du jour D ne changent pas si l'on remplace le futur par du bruit.
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

### Réseau récurrent (GRU)

`python -m src.run_gru` (~8 min : 6 entraînements de ~80 s par pli, plis en parallèle ; nécessite `run_rolling` avant). Un GRU (PyTorch) lit les 14 derniers jours, un pas par jour (les 48 valeurs de la journée en heure locale + son calendrier), et prédit directement les 48 valeurs du lendemain. La consommation est normalisée avec une moyenne et un écart-type calculés **sur les jours d'entraînement seulement** ; le nombre d'époques est choisi sur les 12 mois précédant le test du pli. Deux variantes : le réseau prédit la journée directement, ou seulement son écart à la veille. 3 graines par variante. MAE en MW (moyenne des 3 graines), mêmes lignes pour tous les modèles :

| Pli | Baseline veille | Gradient boosting | GB écart à la veille | GRU | GRU écart à la veille | RTE J-1 |
|---|---|---|---|---|---|---|
| 2021-22 | 2 737 | 1 212 | 1 162 | 794 | 726 | 967 |
| 2022-23 | 2 575 | 1 209 | 1 145 | 824 | 743 | 1 191 |
| 2023-24 | 2 514 | 1 293 | 1 186 | 800 | 734 | 1 369 |
| 2024-25 | 2 536 | 1 344 | 1 243 | 847 | 765 | 1 309 |
| 2025-26 | 2 444 | 1 319 | 1 288 | 847 | 796 | 1 388 |
| **Moyenne** | 2 561 | 1 275 | 1 205 | **822** | **753** | 1 245 |
| Écart-type entre plis | 109 | 62 | 60 | 25 | 28 | 173 |

- Le GRU a une MAE plus basse que tous les autres modèles sur les 5 plis (environ −35 % par rapport au gradient boosting et −68 % par rapport à la baseline « veille », en moyenne sur les plis). La variante « écart à la veille » est meilleure que la variante directe sur les 5 plis.
- **La comparaison avec le gradient boosting n'est pas à information égale.** Le gradient boosting ci-dessus n'utilise que des retards à la même heure, alors que le GRU lit toute la journée de la veille, y compris ses dernières heures avant minuit, qui sont connues à la fois de la règle d'information et de la prévision. Pour mesurer ce qui vient de l'information, un diagnostic (`python -m src.run_gbm_profile`) donne au gradient boosting les 48 valeurs de la veille en plus de ses variables : sa MAE moyenne passe de 1 275 à 975 MW (944 avec la variante « écart à la veille »), contre 784 et 722 MW pour la moyenne des 3 graines du GRU (822 et 753 pour la moyenne des scores de chaque graine). **Plus de la moitié de l'avance initiale du GRU (de 54 % à 66 % selon la variante et selon qu'on prend la moyenne des graines ou des prédictions) vient donc de l'information, pas de l'architecture.** À information égale, le GRU reste meilleur sur les 5 plis (de 16 % à 20 % de MAE en moins en moyenne, avec les scores par graine), sans que l'on sache s'il le doit au réseau récurrent, à la fenêtre de 14 jours ou à la sortie des 48 valeurs d'un coup.
- **Face à RTE, il faut rester prudent** : notre règle d'information donne accès à toute la veille jusqu'à 23h30, ce qui n'est pas forcément le cas de la prévision de RTE (non vérifié). Un meilleur score que RTE ne prouve pas que le modèle est meilleur.
- **Variabilité selon la graine** : pour un même pli, la MAE de la variante directe varie de 1 à 64 MW d'une graine à l'autre (environ 4 % en moyenne). La moyenne des 3 prédictions fait mieux que chaque graine (784 contre 822 MW de moyenne). L'écart entre plis (25 MW) est plus faible que celui du gradient boosting (62 MW).
- Erreur du GRU « écart » (moyenne des 3 prédictions) par heure : environ 240 MW à minuit (juste après la dernière observation) et jusqu'à ~1 000 MW en fin d'après-midi ; par saison : 1 091 MW en hiver contre 448 MW en été.
- Limites propres au GRU : le réglage se limite au nombre d'époques (plafond de 40 atteint dans 10 des 30 entraînements) ; architecture et hyperparamètres (64 unités, pas d'apprentissage 10⁻³, fenêtre de 14 jours) fixés a priori, sans recherche.

## Modèle final et API

**Choix du modèle final.** Le modèle retenu est le **GRU « écart à la veille »**, moyenne de 3 graines. Le critère est la MAE de **validation** (les 12 mois avant chaque pli), pas celle des plis de test : sur les 5 plis, le classement de validation est le même, GRU « écart » < GRU direct < gradient boosting « écart » < gradient boosting (par exemple sur le dernier pli : 743, 798, 1 236 et 1 276 MW). Ce modèle a une MAE moyenne de 722 MW sur les 5 plis de test (moyenne des 3 prédictions). Avant la mise à disposition, il est ré-entraîné sur tout le passé disponible (jusqu'au 31 août 2026), avec le nombre d'époques choisi sur les 12 derniers mois : `python -m src.train_final` (~4 min, écrit `models/gru_final.pt`, non versionné).

**API** (FastAPI) :

```bash
python -m src.train_final                 # une fois : écrit models/gru_final.pt
uvicorn api.main:app                      # lance le serveur (modèle chargé une seule fois au démarrage)
python -m src.example_request             # écrit data/example_request.json à partir des vraies données
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @data/example_request.json
```

- `GET /health` : état du service, version du modèle, date de fin d'entraînement.
- `POST /predict` : le client envoie `date` (le jour à prévoir, heure de Paris) et `history` (consommation par pas de 30 minutes, avec fuseau horaire, au moins 14 jours jusqu'à la veille). L'API renvoie les valeurs prévues du jour (48, ou 46 et 50 les jours de changement d'heure), horodatées en UTC.
- L'API **refuse** (erreur 422, avec un message) un historique qui contient des données du jour à prévoir ou d'après (règle d'information), qui est trop court ou trop incomplet (moins de 95 % de valeurs sur les 14 jours), sans fuseau horaire, hors de la grille de 30 minutes ou avec des valeurs hors de ]0 ; 200 000] MW.
- Le service utilise le même code que l'évaluation (`src/gru.py`) et le modèle est sauvegardé avec sa normalisation : un test vérifie que l'API et le code d'évaluation donnent les mêmes valeurs.
- Essai réel en local (une requête de 768 observations) : réponse en environ 0,15 s.

**Ce qui manque pour une vraie production** : surveillance de la qualité des prévisions (comparaison avec la consommation réalisée) et des données reçues, ré-entraînement planifié, détection de dérive (le niveau de consommation évolue), gestion des données **temps réel** de RTE qui peuvent différer des données définitives sur lesquelles le modèle est entraîné, authentification, limitation du débit, journalisation structurée, versionnement des modèles. Aucun de ces points n'est fait ici.

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
python -m src.run_gru            # réseau récurrent sur les mêmes plis (~8 min), après run_rolling
python -m src.run_gbm_profile    # diagnostic : gradient boosting avec le profil de la veille, après run_gru
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
├── src/             (données, variables, métriques, baselines, modèles, évaluation, service de prévision)
├── api/             (FastAPI : /health et /predict)
├── tests/
├── models/          (modèles sauvegardés, non versionnés)
└── requirements.txt
```
