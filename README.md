# Tarification Assurance Automobile – API de Pricing

## Description

Ce projet consiste en la mise en place d’une API de tarification d’assurance automobile développée avec FastAPI.  
L’objectif est de prédire le montant d’une prime d’assurance à partir des caractéristiques d’un assuré et de son véhicule.

Le calcul repose sur un modèle actuariel classique :

**Prime = Fréquence × Sévérité × α**

- **Fréquence** : probabilité d’occurrence d’un sinistre  
- **Sévérité** : coût estimé du sinistre conditionnellement à sa survenue  
- **α (alpha)** : facteur de calibration  

---

## Fonctionnement

Le système s’appuie sur deux modèles de machine learning :

- un modèle de fréquence (classification)
- un modèle de sévérité (régression)

Le pipeline de traitement est le suivant :

1. Application de transformations métiers (feature engineering)
2. Encodage des variables catégorielles (target encoding)
3. Prédiction de la fréquence de sinistre
4. Prédiction du montant de sinistre
5. Calcul de la prime finale

---

## Structure du projet
induIA/
├── app/
│ ├── main.py # Point d’entrée FastAPI
│ ├── services/
│ │ └── service.py # Logique métier (TarificationEngine)
│ ├── data/
│ │ └── models/ # Modèles sérialisés (.pkl)
│ ├── logs/ # Fichiers de logs
│ └── logging_config.py # Configuration du logging
│
├── tests/
│ ├── test_api.py # Tests des endpoints
│ └── test_service.py # Tests unitaires
│
├── pyproject.toml
├── uv.lock
└── README.md

# Installation

### Clonage du dépôt

bash
git clone https://github.com/RsShanks/induIA.git
cd induIA
code ..
uv sync #gestion des dépendance
uv run uvicorn app.main:app --reload #lancement
API dispo a cette adresse http://127.0.0.1:8000/docs

##ENDPOINT
GET /health
POST /predict_freq #Prédiction de fréquence
POST /predict_sev #Prédiction de srverité
POST /predict_premium #Prédiction de prime

##TEST
Les tests couvrent les endpoints ainsi que la logique métier.
uv run pytest -v

##LOGGING
les logs sont écrits dan sapp/logs/app.log
Ils permettent de tracer :

le démarrage de l’application
le chargement des modèles
les prédictions effectuées
les erreurs éventuelles