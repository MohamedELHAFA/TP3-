# functional_gold.py
#!/usr/bin/env python3
import io, json, boto3, joblib
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuration immuable
CONFIG_GOLD = {
    'endpoint': 'http://localhost:9000',
    'access_key': 'minioadmin',
    'secret_key': 'minioadmin123',
    'buckets': {
        'silver': 'silver',
        'gold': 'gold'
    }
}

def make_s3_client(cfg: Dict[str, Any]):
    return boto3.client(
        's3',
        endpoint_url=cfg['endpoint'],
        aws_access_key_id=cfg['access_key'],
        aws_secret_access_key=cfg['secret_key'],
    )
s3_gold = make_s3_client(CONFIG_GOLD)

def read_features(bucket: str, key: str) -> pd.DataFrame:
    obj = s3_gold.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj['Body'].read()))

def split_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=['sensor_id', 'ts', 'fill_level'])
    y = df['fill_level']
    return X, y

def build_pipeline() -> Pipeline:
    return Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(random_state=42))
    ])

def grid_search_cv(pipe: Pipeline, X, y) -> GridSearchCV:
    param_grid = {
        'rf__n_estimators': [50, 100],
        'rf__max_depth': [None, 10],
        'rf__min_samples_leaf': [1, 2]
    }
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    return GridSearchCV(pipe, param_grid, cv=cv, scoring='neg_root_mean_squared_error')

def evaluate_model(model, X_test, y_test) -> Dict[str, Any]:
    y_pred = model.predict(X_test)
    return {
        'rmse': mean_squared_error(y_test, y_pred, squared=False),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }

def save_to_s3(bucket: str, key: str, data: bytes):
    s3_gold.put_object(Bucket=bucket, Key=key, Body=data)

def main():
    df = read_features(CONFIG_GOLD['buckets']['silver'], 'features/functional_features.parquet')
    X, y = split_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = build_pipeline()
    gs = grid_search_cv(pipe, X_train, y_train)
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_
    metrics = evaluate_model(best_model, X_test, y_test)
    metrics['best_params'] = gs.best_params_

    buf_model = io.BytesIO()
    joblib.dump(best_model, buf_model)
    buf_model.seek(0)
    save_to_s3(CONFIG_GOLD['buckets']['gold'], 'model/functional_model.pkl', buf_model.read())

    metrics_bytes = json.dumps({**metrics, 'timestamp': datetime.utcnow().isoformat()}).encode()
    save_to_s3(CONFIG_GOLD['buckets']['gold'], 'metrics/functional_metrics.json', metrics_bytes)

    print(f"✅ Gold pipeline fonctionnel terminé. Metrics: {metrics}")

if __name__ == '__main__':
    main()
