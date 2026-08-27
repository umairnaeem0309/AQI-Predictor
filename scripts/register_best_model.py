#!/usr/bin/env python
"""Train XGBoost on full training set and register in MLflow."""
import sys, time, json, pickle
sys.path.insert(0, '.')
import numpy as np, pandas as pd
import mlflow, mlflow.sklearn
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from pathlib import Path

TC = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
EX = ["timestamp","location_id","city_name","data_source","aqi_category","aqi_standard","aqi_method","aqi_method_version","aqi_source"]

# Load data
train_f = pd.read_csv("data/processed/train_features.csv")
train_t = pd.read_csv("data/processed/train_targets.csv")
val_f = pd.read_csv("data/processed/val_features.csv")
val_t = pd.read_csv("data/processed/val_targets.csv")
test_f = pd.read_csv("data/processed/test_features.csv")
test_t = pd.read_csv("data/processed/test_targets.csv")

fc = [c for c in train_f.columns if c not in EX and train_f[c].dtype in ["float64","int64","bool"]]

def prep(feat, tgt):
    mask = tgt[TC].notna().all(axis=1)
    return feat.loc[mask, fc].fillna(0).values, tgt.loc[mask, TC].values

X_train, y_train = prep(train_f, train_t)
X_val, y_val = prep(val_f, val_t)
X_test, y_test = prep(test_f, test_t)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Train XGBoost on FULL training set
print("Training XGBoost on full training set...")
t0 = time.time()
model = MultiOutputRegressor(
    xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, verbosity=0, n_jobs=-1,
    )
)
model.fit(X_train, y_train)
train_time = time.time() - t0
print(f"Training complete: {train_time:.1f}s")

# Evaluate on test set
y_pred = model.predict(X_test)

def ev(yt, yp):
    r = {}
    for i, h in enumerate(["24h", "48h", "72h"]):
        r[h] = {"mae": round(mean_absolute_error(yt[:,i], yp[:,i]), 2),
                "rmse": round(np.sqrt(mean_squared_error(yt[:,i], yp[:,i])), 2),
                "r2": round(r2_score(yt[:,i], yp[:,i]), 4)}
    r["overall"] = {
        "mae": round(mean_absolute_error(yt.flatten(), yp.flatten()), 2),
        "rmse": round(np.sqrt(mean_squared_error(yt.flatten(), yp.flatten())), 2),
        "r2": round(r2_score(yt.flatten(), yp.flatten()), 4),
    }
    return r

metrics = ev(y_test, y_pred)
print("\n=== FINAL TEST METRICS ===")
print(f"Overall: MAE={metrics['overall']['mae']}, RMSE={metrics['overall']['rmse']}, R²={metrics['overall']['r2']}")
for h in ["24h", "48h", "72h"]:
    print(f"  {h}: MAE={metrics[h]['mae']}, RMSE={metrics[h]['rmse']}, R²={metrics[h]['r2']}")

# Log to MLflow
mlflow.set_experiment("aqi_predictor_production")
with mlflow.start_run(run_name="XGBoost_production_final"):
    mlflow.log_param("model", "XGBoost_MultiOutput")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("train_rows", len(X_train))
    mlflow.log_param("val_rows", len(X_val))
    mlflow.log_param("test_rows", len(X_test))
    mlflow.log_param("n_features", X_train.shape[1])
    mlflow.log_param("feature_version", "1.0.0")
    mlflow.log_param("dataset_type", "real_api_data")
    mlflow.log_param("data_provider", "open-meteo")
    mlflow.log_param("train_time_s", round(train_time, 1))

    for h in ["24h", "48h", "72h"]:
        mlflow.log_metric(f"test_mae_{h}", metrics[h]["mae"])
        mlflow.log_metric(f"test_rmse_{h}", metrics[h]["rmse"])
        mlflow.log_metric(f"test_r2_{h}", metrics[h]["r2"])
    mlflow.log_metric("test_overall_mae", metrics["overall"]["mae"])
    mlflow.log_metric("test_overall_rmse", metrics["overall"]["rmse"])
    mlflow.log_metric("test_overall_r2", metrics["overall"]["r2"])

    mlflow.sklearn.log_model(model, "model")

    # Save feature columns
    model_info = {
        "feature_columns": fc,
        "target_columns": TC,
        "model_params": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
        "metrics": metrics,
        "train_time": round(train_time, 1),
        "dataset": "real_api_data",
        "data_provider": "open-meteo",
    }
    mlflow.log_dict(model_info, "model_metadata.json")

    run_id = mlflow.active_run().info.run_id
    print(f"\nMLflow run: {run_id}")

# Save model locally
model_dir = Path("models/production")
model_dir.mkdir(parents=True, exist_ok=True)
with open(model_dir / "xgboost_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open(model_dir / "model_metadata.json", "w") as f:
    json.dump(model_info, f, indent=2)
print(f"Model saved to {model_dir}")
