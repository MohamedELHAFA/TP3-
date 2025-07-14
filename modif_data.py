import io
import json
import random
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

import boto3
import pandas as pd
import numpy as np
from botocore.client import Config as BConfig
from dotenv import load_dotenv

# --- Configuration ---
class Config:
    SILVER_BUCKET: str
    GOLD_BUCKET: str
    S3_ENDPOINT: str
    ACCESS_KEY: str
    SECRET_KEY: str

    @staticmethod
    def load_from_env() -> 'Config':
        load_dotenv()
        cfg = Config()
        cfg.S3_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'http://localhost:9000')
        cfg.ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
        cfg.SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin123')
        cfg.SILVER_BUCKET = os.getenv('SILVER_BUCKET', 'silver')
        cfg.GOLD_BUCKET = os.getenv('GOLD_BUCKET', 'gold')
        return cfg

# --- I/O pur ---
def make_s3_client(cfg: Config) -> boto3.client:
    return boto3.client(
        's3',
        endpoint_url=cfg.S3_ENDPOINT,
        aws_access_key_id=cfg.ACCESS_KEY,
        aws_secret_access_key=cfg.SECRET_KEY,
        config=BConfig(signature_version='s3v4')
    )

def ensure_bucket(client: boto3.client, bucket: str) -> None:
    try:
        client.head_bucket(Bucket=bucket)
    except Exception:
        client.create_bucket(Bucket=bucket)


def upload_parquet(client: boto3.client, bucket: str, key: str, df: pd.DataFrame) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    client.put_object(Bucket=bucket, Key=key, Body=buf.read())


def upload_text(client: boto3.client, bucket: str, key: str, text: str) -> None:
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=text.encode('utf-8'),
        ContentType='text/plain'
    )


def upload_json(client: boto3.client, bucket: str, key: str, data: Any) -> None:
    payload = json.dumps(data, indent=2).encode('utf-8')
    client.put_object(Bucket=bucket, Key=key, Body=payload, ContentType='application/json')

# --- Générateurs purs ---
def generate_positions(n: int) -> pd.DataFrame:
    paris_zones = [
        {'lat':48.8566,'lon':2.3522,'w':3},
        {'lat':48.8606,'lon':2.3376,'w':2},
        {'lat':48.8462,'lon':2.3372,'w':2},
    ]
    data = []
    for i in range(n):
        zone = random.choices(paris_zones, weights=[z['w'] for z in paris_zones])[0]
        lat = round(zone['lat'] + random.gauss(0,0.008),6)
        lon = round(zone['lon'] + random.gauss(0,0.012),6)
        data.append({'sensor_id': f'S{i+1}', 'lat': lat, 'lon': lon, 'capacity_tons': 0.12})
    return pd.DataFrame(data)


def generate_history(df_pos: pd.DataFrame, days: int=30) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    start = datetime.now() - timedelta(days=days)
    for day in range(days):
        date = start + timedelta(days=day)
        for hour in range(0,24,2):
            ts = date + timedelta(hours=hour)
            for _, row in df_pos.iterrows():
                base = 0.2 + (hour/24)*0.7
                noise = random.gauss(0,0.08)
                lvl = max(0.0, min(1.0, base + noise))
                rows.append({'sensor_id': row['sensor_id'], 'timestamp': int(ts.timestamp()*1000), 'fill_level': round(lvl,3)})
    df = pd.DataFrame(rows)
    return df[['sensor_id','timestamp','fill_level']]


def generate_tonnage(n: int) -> pd.DataFrame:
    city_total = 146_949_553.36
    data = []
    total = 0.0
    for i in range(n):
        factor = 1.8 if i%15==0 else 1.3 if i%7==0 else 0.8
        var = random.uniform(0.7,1.4)
        ann = (city_total/n) * factor * var
        daily = ann/365
        data.append({'sensor_id': f'S{i+1}', 'annual_tons': round(ann,2), 'daily_tons': round(daily,4)})
        total += ann
    norm = city_total/total
    for d in data:
        d['annual_tons'] = round(d['annual_tons']*norm,2)
        d['daily_tons'] = round(d['daily_tons']*norm,4)
    return pd.DataFrame(data)


def generate_fill_levels(n: int) -> List[int]:
    levels = []
    hour = datetime.now().hour
    for i in range(n):
        base = 30 * (1 + hour/24*0.6)
        factor = 0.7 + (i%5)*0.15
        lvl = base * factor + random.gauss(0,8)
        if i%15==0:
            lvl = random.randint(85,95)
        elif i%7==0:
            lvl = random.randint(70,84)
        levels.append(int(max(15, min(100, lvl))))
    return levels


def create_truck_positions(n: int=5) -> Dict[str, List[List[float]]]:
    depot_zones = [
        {'lat':48.8336,'lon':2.3027},
        {'lat':48.8789,'lon':2.3431},
        {'lat':48.8445,'lon':2.4180},
        {'lat':48.8507,'lon':2.2621},
        {'lat':48.8566,'lon':2.3522},
    ]
    positions = []
    for i in range(min(n, len(depot_zones))):
        zone = depot_zones[i]
        lat = round(zone['lat'] + random.gauss(0,0.002),6)
        lon = round(zone['lon'] + random.gauss(0,0.003),6)
        positions.append([lat, lon])
    return {'positions': positions}

# --- Fonction principale ---
def enhance_all_data():
    """Point d'entrée unique"""
    cfg = Config.load_from_env()
    s3 = make_s3_client(cfg)

    # S'assurer que les buckets existent
    ensure_bucket(s3, cfg.SILVER_BUCKET)
    ensure_bucket(s3, cfg.GOLD_BUCKET)

    # 1. Silver: positions
    df_pos = generate_positions(45)
    upload_parquet(s3, cfg.SILVER_BUCKET, 'sensors/positions.parquet', df_pos)

    # 2. Silver: historique
    df_hist = generate_history(df_pos, days=30)
    upload_parquet(s3, cfg.SILVER_BUCKET, 'sensors/historic_fill.parquet', df_hist)

    # 3. Silver: tonnage
    df_ton = generate_tonnage(45)
    upload_parquet(s3, cfg.SILVER_BUCKET, 'sensor_tonnage/proxy.parquet', df_ton)

    # 4. Gold: niveaux actuels
    levels = generate_fill_levels(45)
    txt = "\n".join(map(str, levels))
    upload_text(s3, cfg.GOLD_BUCKET, 'sensor/sensor_data.txt', txt)

    # 5. Gold: positions capteurs
    sensor_positions = df_pos[['sensor_id','lat','lon']].to_dict(orient='records')
    upload_json(s3, cfg.GOLD_BUCKET, 'sensor/sensor_position.json', {'positions': sensor_positions})

    # 6. Gold: positions camions
    truck_positions = create_truck_positions(5)
    upload_json(s3, cfg.GOLD_BUCKET, 'sensor/home_positions.json', truck_positions)

    print("✅ Données publiées dans Silver & Gold avec succès !")

if __name__ == '__main__':
    enhance_all_data()