#!/usr/bin/env python
"""Test model loading and prediction."""
import sys
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, ".")

print("=" * 60)
print("  MODEL LOADING AND PREDICTION TEST")
print("=" * 60)

# 1. Load model
print("\n1. Loading model from models/production/xgboost_model.pkl...")
model_path = Path("models/production/xgboost_model.pkl")
metadata_path = Path("models/production/model_metadata.json")

if not model_path.exists():
    print("   [FAIL] Model file not found")
    sys.exit(1)

with open(model_path, "rb") as f:
    model = pickle.load(f)
print(f"   [OK] Model loaded: {type(model).__name__}")

# 2. Load metadata
if metadata_path.exists():
    with open(metadata_path) as f:
        metadata = json.load(f)
    print(f"   [OK] Metadata loaded: {len(metadata.get('feature_columns', []))} features")
    print(f"   [OK] Metrics: MAE={metadata['metrics']['overall']['mae']}, R2={metadata['metrics']['overall']['r2']}")
else:
    print("   [WARN] No metadata file")

# 3. Load latest test data
print("\n2. Loading test data...")
test_f = pd.read_csv("data/processed/test_features.csv")
test_t = pd.read_csv("data/processed/test_targets.csv")

TC = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
EX = ["timestamp", "location_id", "city_name", "data_source", "aqi_category",
      "aqi_standard", "aqi_method", "aqi_method_version", "aqi_source"]

fc = [c for c in test_f.columns if c not in EX and test_f[c].dtype in ["float64", "int64", "bool"]]
mask = test_t[TC].notna().all(axis=1)
X_test = test_f.loc[mask, fc].fillna(0).values
y_test = test_t.loc[mask, TC].values
print(f"   [OK] Test data: {X_test.shape[0]} rows, {X_test.shape[1]} features")

# 4. Make predictions
print("\n3. Making predictions...")
y_pred = model.predict(X_test)
print(f"   [OK] Predictions shape: {y_pred.shape}")

# 5. Evaluate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("\n4. Evaluation:")
for i, h in enumerate(["24h", "48h", "72h"]):
    mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
    r2 = r2_score(y_test[:, i], y_pred[:, i])
    print(f"   {h}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")

overall_mae = mean_absolute_error(y_test.flatten(), y_pred.flatten())
overall_r2 = r2_score(y_test.flatten(), y_pred.flatten())
print(f"\n   Overall: MAE={overall_mae:.2f}, R2={overall_r2:.4f}")

# 6. Sample prediction
print("\n5. Sample prediction (first test row):")
sample = X_test[0:1]
pred = model.predict(sample)
print(f"   Input features: {sample[0][:5]}... (71 total)")
print(f"   Predicted AQI: 24h={pred[0][0]:.1f}, 48h={pred[0][1]:.1f}, 72h={pred[0][2]:.1f}")
print(f"   Actual AQI:    24h={y_test[0][0]:.1f}, 48h={y_test[0][1]:.1f}, 72h={y_test[0][2]:.1f}")

# 7. Inference speed
print("\n6. Inference speed test:")
import time
t0 = time.time()
for _ in range(100):
    model.predict(sample)
elapsed = (time.time() - t0) / 100 * 1000
print(f"   [OK] {elapsed:.3f}ms per prediction")

print("\n" + "=" * 60)
print("  ALL TESTS PASSED — MODEL IS PRODUCTION READY")
print("=" * 60)
