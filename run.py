#!/usr/bin/env python3
"""
run_all.py — Orchestrateur pour Smart City Waste Management

Ce script lance successivement :
 1. La création des buckets (via s3clinet.py)
 2. L’ingestion des données brutes (raw_ingest_api.py --once)
 3. Le pipeline Silver ETL (silver.py)
 4. L’entraînement et génération des prédictions (gold.py)
 5. L’amélioration/cohérence des données (modif_data.py)
 6. Le dashboard Streamlit (dashboard.py)
"""

import subprocess
import sys

def main():
    python = sys.executable

    # 1. Création / vérification des buckets S3
    print("✅ Étape 1 : Création des buckets")
    subprocess.run([python, "s3clinet.py"], check=True)

    # 2. Ingestion brute (une seule itération)
    print("✅ Étape 2 : Ingestion RAW")
    subprocess.run([python, "raw_ingest_api.py", "--once"], check=True)

    # 3. Pipeline Silver
    print("✅ Étape 3 : Silver ETL")
    subprocess.run([python, "silver.py"], check=True)

    # 4. Pipeline Gold (une seule passe)


    # 5. Amélioration des données
    print("✅ Étape 5 : Data Enhancement")
    from modif_data import enhance_all_data
    enhance_all_data()

    # 6. Démarrage du dashboard
    print("✅ Étape 6 : Démarrage du Dashboard Streamlit")
    subprocess.run([python, "-m", "streamlit", "run", "dashboard.py"], check=True)

if __name__ == "__main__":
    main()
