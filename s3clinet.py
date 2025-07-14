# s3_client.py

import os
from botocore.client import Config
import boto3

# --- Configuration S3/MinIO depuis variables d’environnement ---
S3_ENDPOINT   = os.getenv("S3_ENDPOINT", "http://localhost:9000")
S3_KEY        = os.getenv("S3_KEY", "minioadmin")
S3_SECRET     = os.getenv("S3_SECRET", "minioadmin123")

RAW_BUCKET    = os.getenv("RAW_BUCKET", "raw")
SILVER_BUCKET = os.getenv("SILVER_BUCKET", "silver")
GOLD_BUCKET   = os.getenv("GOLD_BUCKET", "gold")

# --- Client S3 ---
s3 = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_KEY,
    aws_secret_access_key=S3_SECRET,
    config=Config(signature_version='s3v4'),
    region_name='us-east-1'
)

# --- Création des buckets si nécessaire ---
existing = {b['Name'] for b in s3.list_buckets().get('Buckets', [])}

for bucket in (RAW_BUCKET, SILVER_BUCKET, GOLD_BUCKET):
    if bucket not in existing:
        s3.create_bucket(Bucket=bucket)
        print(f"Bucket créé : {bucket}")
    else:
        print(f"Bucket déjà existant : {bucket}")
