#!/usr/bin/env python
"""
Train All Models — Runs Ridge, Random Forest, XGBoost, and LSTM.

Evaluates each model on test set with MAE, RMSE, R² per horizon.
Logs all experiments to local MLflow.
"""

import sys
import json
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_COLS = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
RANDOM_SEED = 42


def load_data():
    """Load train/val/test splits."""
    train_f = pd.read_csv("data/processed/train_features.csv")
    train_t = pd.read_csv("data/processed/train_targets.csv")
    val_f = pd.read_csv("data/processed/val_features.csv")
    val_t = pd.read_csv("data/processed/val_targets.csv")
    test_f = pd.read_csv("data/processed/test_features.csv")
    test_t = pd.read_csv("data/processed/test_targets.csv")

    exclude = ["timestamp", "location_id", "city_name", "data_source",
               "aqi_category", "aqi_standard", "aqi_method",
               "aqi_method_version", "aqi_source"]
    feature_cols = [c for c in train_f.columns if c not in exclude
                    and train_f[c].dtype in ["float64", "int64", "bool"]]

    def prep(feat, tgt):
        mask = tgt[TARGET_COLS].notna().all(axis=1)
        X = feat.loc[mask, feature_cols].fillna(0).values
        y = tgt.loc[mask, TARGET_COLS].values
        return X, y

    X_train, y_train = prep(train_f, train_t)
    X_val, y_val = prep(val_f, val_t)
    X_test, y_test = prep(test_f, test_t)

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def evaluate(y_true, y_pred):
    """Compute per-horizon and overall metrics."""
    results = {"per_horizon": [], "overall": {}}
    for i, h in enumerate(["24h", "48h", "72h"]):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        results["per_horizon"].append({
            "horizon": h, "mae": round(mae, 2),
            "rmse": round(rmse, 2), "r2": round(r2, 4),
        })
    results["overall"] = {
        "mae": round(mean_absolute_error(y_true.flatten(), y_pred.flatten()), 2),
        "rmse": round(np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten())), 2),
        "r2": round(r2_score(y_true.flatten(), y_pred.flatten()), 4),
    }
    return results


def train_ridge(X_train, y_train, X_test, y_test):
    """Train Ridge regression."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    t0 = time.time()
    model = Ridge(alpha=1.0, random_state=RANDOM_SEED)
    model.fit(X_train_s, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_test_s)
    infer_time = (time.time() - t0) / len(X_test) * 1000

    metrics = evaluate(y_test, y_pred)
    return {
        "model": model, "metrics": metrics,
        "train_time": round(train_time, 3),
        "infer_latency_ms": round(infer_time, 4),
        "params": {"alpha": 1.0},
    }


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest."""
    t0 = time.time()
    model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=100, max_depth=20,
            random_state=RANDOM_SEED, n_jobs=-1,
        )
    )
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_test)
    infer_time = (time.time() - t0) / len(X_test) * 1000

    metrics = evaluate(y_test, y_pred)

    # Feature importance from first estimator
    importance = {}
    if hasattr(model.estimators_[0], "feature_importances_"):
        imp = model.estimators_[0].feature_importances_
        importance = dict(enumerate(imp.tolist()))

    return {
        "model": model, "metrics": metrics,
        "train_time": round(train_time, 1),
        "infer_latency_ms": round(infer_time, 4),
        "params": {"n_estimators": 100, "max_depth": 20},
        "feature_importance_top10": dict(sorted(importance.items(), key=lambda x: -x[1])[:10]),
    }


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost."""
    t0 = time.time()
    model = MultiOutputRegressor(
        xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_SEED, verbosity=0, n_jobs=-1,
        )
    )
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_test)
    infer_time = (time.time() - t0) / len(X_test) * 1000

    metrics = evaluate(y_test, y_pred)

    importance = {}
    if hasattr(model.estimators_[0], "feature_importances_"):
        imp = model.estimators_[0].feature_importances_
        importance = dict(enumerate(imp.tolist()))

    return {
        "model": model, "metrics": metrics,
        "train_time": round(train_time, 1),
        "infer_latency_ms": round(infer_time, 4),
        "params": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
        "feature_importance_top10": dict(sorted(importance.items(), key=lambda x: -x[1])[:10]),
    }


def train_lstm(X_train, y_train, X_test, y_test):
    """Train LSTM model."""
    from src.models.lstm_model import LSTMModel

    seq_len = 24
    n_features = X_train.shape[1]
    n_targets = y_train.shape[1]

    model = LSTMModel(
        sequence_length=seq_len,
        n_features=n_features,
        n_targets=n_targets,
        lstm_units=[64, 32],
        dropout_rate=0.2,
        learning_rate=0.001,
        random_seed=RANDOM_SEED,
    )

    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        X_test, y_test,
        epochs=50, batch_size=32, verbose=0,
    )
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_test)
    infer_time = (time.time() - t0) / max(len(y_pred), 1) * 1000

    # Align: LSTM predictions start at seq_len-1
    n_pred = len(y_pred)
    y_test_aligned = y_test[seq_len - 1:seq_len - 1 + n_pred]

    metrics = evaluate(y_test_aligned, y_pred)

    return {
        "model": model, "metrics": metrics,
        "train_time": round(train_time, 1),
        "infer_latency_ms": round(infer_time, 4),
        "params": {
            "sequence_length": seq_len,
            "lstm_units": [64, 32],
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "epochs": 50,
            "batch_size": 32,
        },
        "epochs_trained": history.get("epochs_trained", 0),
    }


def main():
    """Run all model experiments and log to MLflow."""
    logger.info("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_data()
    logger.info(
        "Data loaded: train=%s, val=%s, test=%s, features=%d",
        X_train.shape, X_val.shape, X_test.shape, len(feature_cols),
    )

    mlflow.set_experiment("aqi_predictor_model_comparison")

    all_results = []

    # --- Ridge ---
    logger.info("=" * 60)
    logger.info("Training Ridge Regression...")
    result = train_ridge(X_train, y_train, X_test, y_test)
    result["model_name"] = "Ridge"
    all_results.append(result)
    logger.info("  MAE: %s, R²: %s", result["metrics"]["overall"]["mae"], result["metrics"]["overall"]["r2"])

    with mlflow.start_run(run_name="Ridge"):
        mlflow.log_param("model", "Ridge")
        mlflow.log_param("alpha", 1.0)
        mlflow.log_param("train_time_s", result["train_time"])
        for h in result["metrics"]["per_horizon"]:
            mlflow.log_metric(f"mae_{h['horizon']}", h["mae"])
            mlflow.log_metric(f"rmse_{h['horizon']}", h["rmse"])
            mlflow.log_metric(f"r2_{h['horizon']}", h["r2"])
        mlflow.log_metric("overall_mae", result["metrics"]["overall"]["mae"])
        mlflow.log_metric("overall_r2", result["metrics"]["overall"]["r2"])
        mlflow.sklearn.log_model(result["model"], "model")

    # --- Random Forest ---
    logger.info("=" * 60)
    logger.info("Training Random Forest...")
    result = train_random_forest(X_train, y_train, X_test, y_test)
    result["model_name"] = "RandomForest"
    all_results.append(result)
    logger.info("  MAE: %s, R²: %s", result["metrics"]["overall"]["mae"], result["metrics"]["overall"]["r2"])

    with mlflow.start_run(run_name="RandomForest"):
        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 20)
        mlflow.log_param("train_time_s", result["train_time"])
        for h in result["metrics"]["per_horizon"]:
            mlflow.log_metric(f"mae_{h['horizon']}", h["mae"])
            mlflow.log_metric(f"rmse_{h['horizon']}", h["rmse"])
            mlflow.log_metric(f"r2_{h['horizon']}", h["r2"])
        mlflow.log_metric("overall_mae", result["metrics"]["overall"]["mae"])
        mlflow.log_metric("overall_r2", result["metrics"]["overall"]["r2"])

    # --- XGBoost ---
    logger.info("=" * 60)
    logger.info("Training XGBoost...")
    result = train_xgboost(X_train, y_train, X_test, y_test)
    result["model_name"] = "XGBoost"
    all_results.append(result)
    logger.info("  MAE: %s, R²: %s", result["metrics"]["overall"]["mae"], result["metrics"]["overall"]["r2"])

    with mlflow.start_run(run_name="XGBoost"):
        mlflow.log_param("model", "XGBoost")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("train_time_s", result["train_time"])
        for h in result["metrics"]["per_horizon"]:
            mlflow.log_metric(f"mae_{h['horizon']}", h["mae"])
            mlflow.log_metric(f"rmse_{h['horizon']}", h["rmse"])
            mlflow.log_metric(f"r2_{h['horizon']}", h["r2"])
        mlflow.log_metric("overall_mae", result["metrics"]["overall"]["mae"])
        mlflow.log_metric("overall_r2", result["metrics"]["overall"]["r2"])

    # --- LSTM ---
    logger.info("=" * 60)
    logger.info("Training LSTM...")
    result = train_lstm(X_train, y_train, X_test, y_test)
    result["model_name"] = "LSTM"
    all_results.append(result)
    logger.info("  MAE: %s, R²: %s", result["metrics"]["overall"]["mae"], result["metrics"]["overall"]["r2"])

    with mlflow.start_run(run_name="LSTM"):
        mlflow.log_param("model", "LSTM")
        mlflow.log_param("sequence_length", 24)
        mlflow.log_param("lstm_units", "[64, 32]")
        mlflow.log_param("epochs", 50)
        mlflow.log_param("train_time_s", result["train_time"])
        for h in result["metrics"]["per_horizon"]:
            mlflow.log_metric(f"mae_{h['horizon']}", h["mae"])
            mlflow.log_metric(f"rmse_{h['horizon']}", h["rmse"])
            mlflow.log_metric(f"r2_{h['horizon']}", h["r2"])
        mlflow.log_metric("overall_mae", result["metrics"]["overall"]["mae"])
        mlflow.log_metric("overall_r2", result["metrics"]["overall"]["r2"])

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)

    rows = []
    for r in all_results:
        for h in r["metrics"]["per_horizon"]:
            rows.append({
                "Model": r["model_name"],
                "Horizon": h["horizon"],
                "MAE": h["mae"],
                "RMSE": h["rmse"],
                "R²": h["r2"],
                "Train Time (s)": r["train_time"],
                "Infer Latency (ms)": r.get("infer_latency_ms", 0),
            })

    summary = pd.DataFrame(rows)
    print("\n" + summary.to_string(index=False))

    # Overall comparison
    print("\n--- Overall ---")
    for r in all_results:
        o = r["metrics"]["overall"]
        print(f"  {r['model_name']:15s}  MAE={o['mae']:6.2f}  RMSE={o['rmse']:6.2f}  R²={o['r2']:.4f}  Time={r['train_time']:.1f}s")

    # Save results
    output = {
        "results": [
            {
                "model": r["model_name"],
                "metrics": r["metrics"],
                "train_time": r["train_time"],
                "infer_latency_ms": r.get("infer_latency_ms", 0),
                "params": r["params"],
            }
            for r in all_results
        ]
    }
    with open("data/processed/model_comparison_results.json", "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to data/processed/model_comparison_results.json")


if __name__ == "__main__":
    main()
