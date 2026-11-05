# conso-forecast

Prévision de la consommation électrique française à J+1, avec comparaison de plusieurs approches et mise à disposition du modèle via une API conteneurisée.

> **Statut : en cours.** Ce README sera complété à l'étape 11, uniquement avec des résultats réellement obtenus.

## Problème

À rédiger : le problème métier, l'horizon de prévision, la métrique de décision.

## Données

À rédiger : source (RTE, éCO2mix), période, pas de temps, licence, traitement des anomalies.

## Méthode

À rédiger : baselines, modèles comparés, variables, validation en origine glissante.

## Résultats

À rédiger : tableau issu des exécutions réelles (modèle, MAE, RMSE, MAPE, moyenne et dispersion sur les plis).

## Limites

À rédiger : événements exceptionnels, dérive des données, ce qui manque pour la production.

## Installation et lancement

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Les commandes pour télécharger les données, entraîner et lancer l'API seront ajoutées au fil des étapes.

## Structure du dépôt

```
conso-forecast/
├── data/            (non versionné : voir src/data.py pour le téléchargement)
├── notebooks/       (exploration)
├── src/             (données, variables, modèles, évaluation)
├── api/             (FastAPI)
├── tests/
├── docs/            (métier, journal de bord, préparation d'entretien)
├── models/          (modèles sauvegardés, non versionnés)
├── CLAUDE.md
├── PLAN.md
└── README.md
```
