# functional_silver.py
#!/usr/bin/env python3
from datetime import datetime, timedelta
from typing import List, Dict, Callable, Any
import boto3, io, json, pandas as pd

# Configuration immuable
CONFIG = {
    'endpoint': 'http://localhost:9000',
    'access_key': 'minioadmin',
    'secret_key': 'minioadmin123',
    'buckets': {
        'raw': 'raw',
        'silver': 'silver'
    }
}

# Effets de bord isolés: client S3
def make_s3_client(cfg: Dict[str, Any]):
    return boto3.client(
        's3',
        endpoint_url=cfg['endpoint'],
        aws_access_key_id=cfg['access_key'],
        aws_secret_access_key=cfg['secret_key']
    )
s3 = make_s3_client(CONFIG)

# 0) Ingestion des positions depuis RAW JSON → Silver parquet
def ingest_positions():
    obj = s3.get_object(Bucket=CONFIG['buckets']['raw'], Key='sensor/sensor_position.json')
    positions = json.loads(obj['Body'].read().decode())['positions']
    df_pos = pd.DataFrame({
        'sensor_id': [f"S{i+1}" for i in range(len(positions))],
        'lat': [p[0] for p in positions],
        'lon': [p[1] for p in positions],
        'capacity_tons': [1.0]*len(positions)
    })
    buf = io.BytesIO()
    df_pos.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=CONFIG['buckets']['silver'], Key='sensors/positions.parquet', Body=buf.read())
    return df_pos

# Lecture JSONL pur
def read_jsonl_records(bucket: str, prefix: str) -> List[Dict]:
    objs = s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get('Contents', [])
    blocks = [
        s3.get_object(Bucket=bucket, Key=o['Key'])['Body'].read().decode()
        for o in objs if o['Key'].endswith('.jsonl')
    ]
    return [json.loads(line) for block in blocks for line in block.splitlines()]

# Transformations pures
def parse_record(r: Dict) -> Dict:
    return {
        'sensor_id': r['sensor_id'],
        'ts': datetime.fromtimestamp(r['timestamp']/1000),
        'fill_level': r['fill_level']
    }
def filter_valid(r: Dict) -> bool:
    return 0.0 <= r['fill_level'] <= 1.0
def enrich_capacity(pos_map: Dict[str, float]) -> Callable[[Dict], Dict]:
    return lambda r: {
        **r,
        'capacity_tons': pos_map.get(r['sensor_id'], 1.0),
        'occupancy_tons': r['fill_level'] * pos_map.get(r['sensor_id'], 1.0)
    }

# Agrégation rolling
def window_aggregate(records: List[Dict], window: timedelta) -> List[Dict]:
    def agg(r: Dict) -> Dict:
        slice_ = [x['fill_level'] for x in records
                  if x['sensor_id']==r['sensor_id']
                  and r['ts'] - window <= x['ts'] <= r['ts']]
        mean = sum(slice_)/len(slice_) if slice_ else None
        std = (sum((x-mean)**2 for x in slice_)/len(slice_))**0.5 if slice_ else None
        return {**r, 'mean_1h': mean, 'std_1h': std}
    return list(map(agg, records))

def functional_silver():
    df_pos = ingest_positions()
    pos_map = dict(zip(df_pos['sensor_id'], df_pos['capacity_tons']))

    records = read_jsonl_records(CONFIG['buckets']['raw'], 'api/colonnes_verre')
    parsed = list(map(parse_record, records))
    filtered = list(filter(filter_valid, parsed))

    enriched = list(map(enrich_capacity(pos_map), filtered))
    sorted_records = sorted(enriched, key=lambda r: (r['sensor_id'], r['ts']))
    with_windows = window_aggregate(sorted_records, timedelta(hours=1))

    df = pd.DataFrame.from_records(with_windows)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=CONFIG['buckets']['silver'], Key='features/functional_features.parquet', Body=buf.read())
    print("✅ Silver pipeline fonctionnel terminé.")

if __name__ == '__main__':
    functional_silver()
