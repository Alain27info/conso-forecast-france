# conso-forecast

Prévision de la consommation électrique française à J+1, avec comparaison de plusieurs approches (baselines, arbres boostés, SARIMA, réseau récurrent) et mise à disposition du meilleur modèle via une API conteneurisée.

> **Statut : en cours de développement.** Les sections ci-dessous ne contiennent que des résultats réellement obtenus et reproductibles. Les autres modèles (SARIMA, réseau récurrent), la validation en origine glissante, l'API et Docker ne sont pas encore faits.

## Problème

Prévoir la consommation nationale d'électricité pour chaque pas de 30 minutes du lendemain. Sur un réseau électrique, la production doit égaler la consommation à chaque instant : une bonne prévision permet de préparer la production à l'avance.

## Données

- **Source** : RTE, *Données éCO2mix nationales consolidées et définitives* (open data), [Licence Ouverte v2.0 (Etalab)](https://www.etalab.gouv.fr/wp-content/uploads/2017/04/ETALAB-Licence-Ouverte-v2.0.pdf).
- **Période** : du 1er janvier 2012 au 31 août 2026 (heure de Paris), 257 134 créneaux de 30 minutes. Les données sont *définitives* jusqu'au 31 décembre 2025, *consolidées* ensuite.
- **Traitement** : index en UTC ; 60 horodatages dupliqués au changement d'heure du printemps (moyennés) ; une heure manquante chaque automne (laissée vide, pas interpolée, pour éviter d'utiliser une valeur du futur). Voir `src/data.py` et `notebooks/01_exploration.ipynb`.
- Les données brutes ne sont pas versionnées : `python -m src.data` les télécharge.

## Méthode

- **Test** : les 12 derniers mois (du 1er septembre 2025 au 31 août 2026). Pas de découpage aléatoire.
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

## Limites

- Un seul découpage train/test pour l'instant ; une validation en origine glissante est prévue.
- Pas de météo : la température, qui influence fortement la consommation (chauffage), n'est pas utilisée.
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
