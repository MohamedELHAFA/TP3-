﻿# TP3-
# Smart Waste Management – Paris

## Description

Ce projet implémente une solution de **Smart Waste Management** pour la ville de Paris, à travers un pipeline fonctionnel complet :

1. **Génération & publication des données** (`modif_data.py`) :

   * Positions géolocalisées des capteurs
   * Historique de remplissage (30 jours, relevé toutes les 2 heures)
   * Tonnage annuel et journalier
   * Niveaux actuels de remplissage
   * Positions des camions de collecte
2. **Stockage Silver** : stockage brut des Parquet dans MinIO (bucket `silver`).
3. **Traitements purs** (`silver.py`) : parsing, filtrage, enrichissement, agrégation glissante (rolling window) pour produire des features.
4. **Orchestration complète** (`run.py`) : exécution séquencée des étapes Silver et Gold.
5. **Pipeline Gold & ML** (`gold.py`) : publication des niveaux actuels, entraînement et validation d’un modèle RandomForest avec GridSearchCV, sauvegarde du modèle et des métriques dans MinIO (bucket `gold`).
6. **Dashboard Streamlit** (`dashboard.py`) : interface interactive pour visualiser la carte des capteurs, les séries temporelles, les KPI et piloter les tournées.

## Prérequis

* **Python 3.10+**
* **Docker & Docker Compose**
* **MinIO** (compatible S3)

## Installation

1. **Cloner le dépôt**

   ```bash
   git clone https://github.com/MohamedELHAFA/TP3-
   cd TP3-
   ```

3. **Démarrer MinIO**

   ```bash
   docker-compose up -d
   ```
4. **Installer les dépendances Python**

   ```bash
   pip install -r requirements.txt
   ```

## Utilisation
   ```bash
python run.py
```



> **Remarque** : `run.py` appelle séquentiellement `silver.py` puis `gold.py`. Veillez à ce que MinIO soit opérationnel via Docker Compose.

## Tests

Des tests unitaires sont disponibles pour les générateurs purs :

```bash
pytest --maxfail=1 --disable-warnings -q
```

## Structure du projet

```
├── modif_data.py       # Génération et publication de données Silver & Gold
├── silver.py           # Pipeline Silver (traitements purs)
├── gold.py             # Pipeline Gold (ML & publication)
├── run.py              # Orchestration complète du pipeline
├── dashboard.py        # Interface Streamlit
├── requirements.txt    # Dépendances Python
├── docker-compose.yaml # Service MinIO
├── .env.example        # Modèle de fichier d’environnement
└── tests/              # Tests pytest pour les générateurs
```

## Contribution

Les contributions sont les bienvenues ! Ouvrez une *issue* ou proposez un *pull request*.

## Licence

Ce projet est sous licence MIT.
