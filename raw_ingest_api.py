#!/usr/bin/env python3
"""
raw_ingest_api.py

Ingestion batch minimale des données brutes indispensables pour
entraîner le modèle de prédiction de niveau de remplissage.

– Toutes les sources sont récupérées via API ou générées en mémoire :
    • sensor_position.json : grille de 45 capteurs créée dynamiquement
    • historic_fill_levels.jsonl : simulation des niveaux passés
    • populations_legales_2021.csv : upload automatique si présent dans raw/demographics/

– Jeux de points d’apport volontaire (GeoJSON via CKAN API)
  colonnes_verre, composteurs, conteneur_textile, stations_trilib, recycleries.
– Tonnage de déchets par habitant (GeoJSON via CKAN API)

Tous les fichiers sont poussés dans le bucket S3 RAW_BUCKET,
avec partitionnement par date et idempotence (skip si déjà existant).
"""

import os
import time
import json
import logging
import requests
import random
from datetime import datetime, timedelta
from botocore.exceptions import ClientError
from s3clinet import s3, RAW_BUCKET

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
API_SOURCES = [
    {"name": "colonnes_verre",   "dataset": "dechets-menagers-points-dapport-volontaire-colonnes-a-verre"},
    {"name": "composteurs",      "dataset": "dechets-menagers-points-dapport-volontaire-composteurs"},
    {"name": "conteneur_textile","dataset": "dechets-menagers-points-dapport-volontaire-conteneur-textile"},
    {"name": "stations_trilib",  "dataset": "dechets-menagers-points-dapport-volontaire-stations-trilib"},
    {"name": "recycleries",      "dataset": "dechets-menagers-points-dapport-volontaire-recycleries-et-ressourceries"},
    {"name": "tonnage_par_habitant", "dataset": "quantite-de-dechets-produits-et-tries-par-habitant-et-par-an"}
]

DEFAULT_INTERVAL = 24 * 3600

# ----------------------------------------------------------------------
# Utilitaires S3 et HTTP
# ----------------------------------------------------------------------
def setup_logging(level="INFO"):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )

def object_exists(key):
    try:
        s3.head_object(Bucket=RAW_BUCKET, Key=key)
        return True
    except ClientError as e:
        if e.response['Error']['Code']=='404': return False
        raise

def put_bytes(key, data, content_type="application/json"):
    if object_exists(key):
        logging.info("Skipped existing s3://%s/%s", RAW_BUCKET, key)
        return
    s3.put_object(Bucket=RAW_BUCKET, Key=key, Body=data, ContentType=content_type)
    logging.info("Uploaded s3://%s/%s", RAW_BUCKET, key)

def fetch_ckan(dataset):
    url = f"https://opendata.paris.fr/api/records/1.0/search/?dataset={dataset}&rows=-1&format=geojson"
    resp = requests.get(url,timeout=30)
    resp.raise_for_status()
    return resp.content

# ----------------------------------------------------------------------
# Génération / ingestion des données brutes
# ----------------------------------------------------------------------
def ingest_sensor_positions(ts):
    base_lat, base_lon = 48.8466, 2.3322
    positions = [[round(base_lat + i*0.005,6), round(base_lon + j*0.005,6)]
                 for i in range(5) for j in range(9)]
    data = json.dumps({"positions": positions}).encode('utf-8')
    put_bytes("sensor/sensor_position.json", data)


def ingest_historic_fill(ts):
    sensors = [f"S{i+1}" for i in range(45)]
    start = ts - timedelta(days=30)
    lines = []
    for h in range(30*24):
        ts_h = int((start + timedelta(hours=h)).timestamp()*1000)
        for s in sensors:
            base = 0.5 + 0.3 * max(0, __import__('math').sin(2*__import__('math').pi*(h%24)/24))
            lvl = min(1.0, max(0.0, base + random.gauss(0,0.05)))
            lines.append(json.dumps({"sensor_id":s,"timestamp":ts_h,"fill_level":lvl}))
    data = "\n".join(lines).encode('utf-8')
    key = ts.strftime("sensor/historic_fill_levels/date=%Y/%m/%d/historic_%Y%m%dT%H%M%SZ.jsonl")
    put_bytes(key, data)


def ingest_population(ts):
    # Upload du CSV populations légales s'il existe dans le répertoire projet
    # On construit le chemin absolu relatif au fichier script
    project_dir = os.path.dirname(os.path.abspath(__file__))
    local_csv = os.path.join(project_dir, "raw", "demographics", "populations_legales_2021.csv")
    s3_key = "demographics/populations_legales_2021.csv"
    if os.path.exists(local_csv):
        with open(local_csv, 'rb') as f:
            put_bytes(s3_key, f.read(), content_type="text/csv")
    else:
        logging.warning(
            "CSV population absent (%s). Télécharger depuis INSEE et placer ici.", local_csv
        )
    local_csv = "raw/demographics/populations_legales_2021.csv"
    s3_key = "demographics/populations_legales_2021.csv"
    if os.path.exists(local_csv):
        with open(local_csv, 'rb') as f:
            put_bytes(s3_key, f.read(), content_type="text/csv")
    else:
        logging.warning(
            "CSV population absent (%s). Télécharger depuis INSEE et placer ici.", local_csv
        )


def ingest_apis(ts):
    for src in API_SOURCES:
        try:
            content = fetch_ckan(src['dataset'])
            ext = '.geojson'
            key = f"api/{src['name']}/date={ts.strftime('%Y/%m/%d')}/{src['name']}_{ts.strftime('%Y%m%dT%H%M%SZ')}{ext}"
            put_bytes(key, content, content_type="application/geo+json")
        except Exception as e:
            logging.warning("Échec ingestion %s: %s", src['name'], e)


def ingest_all():
    ts = datetime.utcnow()
    ingest_sensor_positions(ts)
    ingest_historic_fill(ts)
    ingest_population(ts)
    ingest_apis(ts)

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=int, default=DEFAULT_INTERVAL)
    parser.add_argument('--log', default='INFO')
    args = parser.parse_args()

    setup_logging(args.log)
    logging.info("Starting ingestion (once=%s interval=%s)", args.once, args.interval)
    if args.once:
        ingest_all()
    else:
        while True:
            ingest_all()
            time.sleep(args.interval)

if __name__=='__main__':
    main()
