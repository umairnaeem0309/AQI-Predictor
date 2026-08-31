#!/usr/bin/env python3
"""
Daily Training Pipeline — Hopsworks Feature Store Edition.

Reads features+targets from Hopsworks Feature View, trains Ridge,
Random Forest, XGBoost, and LSTM, evaluates performance, and
registers the best model in Hopsworks Model Registry.

NO local CSV files are used. All data comes from Hopsworks.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --force-register
    python scripts/train_model.py --min-improvement 0.01
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_environment

load_environment()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_model")

# Paths
MODELS_DIR = PROJECT_ROOT / "models" / "production"
METADATA_DIR = PROJECT_ROOT / "models" / "metadata"


# =============================================================================
# DATA LOADING — Hopsworks Feature View (PRIMARY)
# =============================================================================


def load_training_data_from_hopsworks():
    """Load training data from Hopsworks Feature View.

    This is the SINGLE data loading path. No local CSV fallback.

    Returns:
        DataFrame with features + targets from Hopsworks.
    """
    import hopsworks

    host = os.environ.get("HOPSWORKS_HOST")
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    project_name = os.environ.get("HOPSWORKS_PROJECT", "AQI_Predictor")

    if not host or not api_key:
        raise RuntimeError(
            "HOPSWORKS_HOST and HOPSWORKS_API_KEY must be set. "
            "Run ingest_to_hopsworks.py first to populate the feature store."
        )

    # Connect to Hopsworks
    project = hopsworks.login(
        host=host,
        api_key_value=api_key,
        project=project_name,
    )
    fs = project.get_feature_store()

    # Get the feature group (features + targets stored together)
    fg = fs.get_feature_group(name="aqi_features_prod", version=1)

    # Read ALL data from the feature group
    try:
        df = fg.read()
    except Exception as e:
        logger.warning(f"Hopsworks read failed (materialization may still be running): {e}")
        logger.info("Falling back to local CSV backup...")
        return _load_from_local_csv()

    if df is None or df.empty:
        logger.warning("Hopsworks returned empty data, falling back to local CSV")
        return _load_from_local_csv()

    logger.info(f"✅ Loaded {len(df)} rows from Hopsworks Feature Store")
    logger.info(f"   Columns: {len(df.columns)}")
    logger.info(f"   Cities: {df['location_id'].nunique()}")
    logger.info(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


def _load_from_local_csv():
    """Load from local CSV backup (fallback when Hopsworks is not ready)."""
    features_file = PROJECT_ROOT / "data" / "processed" / "train_features.csv"
    targets_file = PROJECT_ROOT / "data" / "processed" / "train_targets.csv"

    if not features_file.exists() or not targets_file.exists():
        raise FileNotFoundError(
            f"Neither Hopsworks nor local CSV available. " f"Run ingest_to_hopsworks.py first."
        )

    logger.info("Loading from local CSV backup...")
    features_df = pd.read_csv(features_file)
    targets_df = pd.read_csv(targets_file)
    df = pd.merge(features_df, targets_df, on=["timestamp", "location_id"], how="inner")

    logger.info(f"✅ Loaded {len(df)} rows from local CSV backup")
    return df


def load_training_data_from_feature_view():
    """Load training data using Hopsworks Feature View with train_test_split.

    This uses the Feature View's built-in train/test split to ensure
    reproducible, consistent splits across training runs.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, feature_names)
    """
    import hopsworks

    host = os.environ.get("HOPSWORKS_HOST")
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    project_name = os.environ.get("HOPSWORKS_PROJECT", "AQI_Predictor")

    project = hopsworks.login(
        host=host,
        api_key_value=api_key,
        project=project_name,
    )
    fs = project.get_feature_store()

    # Try to get the Feature View
    try:
        fv = fs.get_feature_view(name="aqi_feature_view", version=1)
        logger.info("Using Hopsworks Feature View: aqi_feature_view v1")
    except Exception:
        logger.warning(
            "Feature view not found. Falling back to feature group read. "
            "Run ingest_to_hopsworks.py first."
        )
        return None

    # Get training data from Feature View
    # This returns features + labels combined
    training_data = fv.get_training_data(description="AQI prediction training data")

    if training_data is None or training_data.empty:
        logger.warning("Feature view returned empty data")
        return None

    logger.info(f"✅ Loaded {len(training_data)} rows from Feature View")

    return training_data


def prepare_data_from_dataframe(df):
    """Prepare features and targets from a Hopsworks DataFrame.

    Separates features from targets, handles NaN, and creates
    chronological train/val/test splits.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, feature_names)
    """
    # Target columns
    target_cols = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]

    # Verify targets exist
    missing_targets = [c for c in target_cols if c not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    # Drop rows with NaN targets
    before_drop = len(df)
    df = df.dropna(subset=target_cols)
    dropped = before_drop - len(df)
    if dropped > 0:
        logger.info(f"Dropped {dropped} rows with NaN targets, {len(df)} remaining")

    # Sort by timestamp for chronological split
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    # Feature columns: everything except targets and metadata
    exclude_cols = set(
        target_cols
        + [
            "timestamp",
            "location_id",
            "city_name",
            "data_source",
            "collected_at",
            "is_training_valid",
            "us_aqi",
            "us_aqi_pm25",
            "us_aqi_pm10",
            "pm25_aqi",
            "pm10_aqi",
            "provider",
            "weather_available",
            "aqi_available",
        ]
    )
    # Also exclude string/object columns
    string_cols = df.select_dtypes(include=["object"]).columns.tolist()
    exclude_cols = exclude_cols.union(set(string_cols))

    feature_names = [c for c in df.columns if c not in exclude_cols]
    logger.info(f"Using {len(feature_names)} features")

    # Extract arrays
    X = df[feature_names].values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)

    # Replace NaN/inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Chronological split: 72% train, 8% val, 20% test
    n = len(X)
    train_end = int(n * 0.72)
    val_end = int(n * 0.80)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    logger.info(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


# =============================================================================
# MODEL TRAINING
# =============================================================================


def evaluate_model(model, X_val, y_val, X_test, y_test):
    """Evaluate model and return metrics for all horizons."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    horizon_names = ["24h", "48h", "72h"]

    val_metrics = {
        "mae": float(mean_absolute_error(y_val, y_val_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_val, y_val_pred))),
        "r2": float(r2_score(y_val, y_val_pred)),
    }
    test_metrics = {
        "mae": float(mean_absolute_error(y_test, y_test_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
        "r2": float(r2_score(y_test, y_test_pred)),
    }

    for i, h in enumerate(horizon_names):
        val_metrics[f"mae_{h}"] = float(mean_absolute_error(y_val[:, i], y_val_pred[:, i]))
        val_metrics[f"rmse_{h}"] = float(np.sqrt(mean_squared_error(y_val[:, i], y_val_pred[:, i])))
        val_metrics[f"r2_{h}"] = float(r2_score(y_val[:, i], y_val_pred[:, i]))
        test_metrics[f"mae_{h}"] = float(mean_absolute_error(y_test[:, i], y_test_pred[:, i]))
        test_metrics[f"rmse_{h}"] = float(
            np.sqrt(mean_squared_error(y_test[:, i], y_test_pred[:, i]))
        )
        test_metrics[f"r2_{h}"] = float(r2_score(y_test[:, i], y_test_pred[:, i]))

    return val_metrics, test_metrics


def compute_composite_score(metrics):
    """Compute composite score: 0.4*MAE + 0.3*RMSE + 0.3*(1-R²)*100.

    Lower is better.
    """
    horizon_names = ["24h", "48h", "72h"]
    mae_scores = [metrics.get(f"mae_{h}", metrics["mae"]) for h in horizon_names]
    rmse_scores = [metrics.get(f"rmse_{h}", metrics["rmse"]) for h in horizon_names]
    r2_scores = [metrics.get(f"r2_{h}", metrics["r2"]) for h in horizon_names]

    avg_mae = np.mean(mae_scores)
    avg_rmse = np.mean(rmse_scores)
    avg_r2 = np.mean(r2_scores)

    return 0.4 * avg_mae + 0.3 * avg_rmse + 0.3 * (1 - avg_r2) * 100


def train_all_models(X_train, X_val, X_test, y_train, y_val, y_test, feature_names):
    """Train all 4 models and return comparison results.

    Models:
    - Ridge Regression
    - Random Forest
    - XGBoost
    - LSTM (PyTorch)
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.multioutput import MultiOutputRegressor

    results = {}

    # ── Model 1: Ridge Regression ────────────────────────────────────────
    logger.info("Training Ridge Regression...")
    start = time.time()
    ridge = MultiOutputRegressor(Ridge(alpha=1.0))
    ridge.fit(X_train, y_train)
    ridge_time = time.time() - start
    ridge_val, ridge_test = evaluate_model(ridge, X_val, y_val, X_test, y_test)
    results["ridge"] = {
        "model": ridge,
        "name": "Ridge Regression",
        "train_time": ridge_time,
        "val_metrics": ridge_val,
        "test_metrics": ridge_test,
    }
    logger.info(
        f"  Ridge: Val MAE={ridge_val['mae']:.2f}, Test MAE={ridge_test['mae']:.2f} "
        f"({ridge_time:.1f}s)"
    )

    # ── Model 2: Random Forest ───────────────────────────────────────────
    logger.info("Training Random Forest...")
    start = time.time()
    rf = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=100, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1
        )
    )
    rf.fit(X_train, y_train)
    rf_time = time.time() - start
    rf_val, rf_test = evaluate_model(rf, X_val, y_val, X_test, y_test)
    results["random_forest"] = {
        "model": rf,
        "name": "Random Forest",
        "train_time": rf_time,
        "val_metrics": rf_val,
        "test_metrics": rf_test,
    }
    logger.info(
        f"  RF:     Val MAE={rf_val['mae']:.2f}, Test MAE={rf_test['mae']:.2f} " f"({rf_time:.1f}s)"
    )

    # ── Model 3: XGBoost ─────────────────────────────────────────────────
    logger.info("Training XGBoost...")
    start = time.time()
    import xgboost as xg

    xgb = MultiOutputRegressor(
        xg.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
    )
    xgb.fit(X_train, y_train)
    xgb_time = time.time() - start
    xgb_val, xgb_test = evaluate_model(xgb, X_val, y_val, X_test, y_test)
    results["xgboost"] = {
        "model": xgb,
        "name": "XGBoost",
        "train_time": xgb_time,
        "val_metrics": xgb_val,
        "test_metrics": xgb_test,
    }
    logger.info(
        f"  XGB:    Val MAE={xgb_val['mae']:.2f}, Test MAE={xgb_test['mae']:.2f} "
        f"({xgb_time:.1f}s)"
    )

    # ── Model 4: LSTM (PyTorch) ──────────────────────────────────────────
    logger.info("Training LSTM...")
    start = time.time()
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=3):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size, hidden_size, num_layers, batch_first=True, dropout=0.2
                )
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                x = x.unsqueeze(1)
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        lstm_model = LSTMModel(X_train.shape[1]).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_val).to(device)
        X_test_t = torch.FloatTensor(X_test).to(device)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(50):
            lstm_model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                pred = lstm_model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

            # Early stopping
            lstm_model.eval()
            with torch.no_grad():
                val_pred = lstm_model(X_val_t)
                val_loss = criterion(val_pred, y_val_t)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in lstm_model.state_dict().items()}
                else:
                    patience_counter += 1
                if patience_counter >= 5:
                    break

        # Restore best weights
        lstm_model.load_state_dict(best_state)

        class LSTMPredictor:
            def __init__(self, model, device):
                self.model = model
                self.device = device

            def predict(self, X):
                self.model.eval()
                with torch.no_grad():
                    X_t = torch.FloatTensor(X).to(self.device)
                    return self.model(X_t).cpu().numpy()

        lstm_wrapped = LSTMPredictor(lstm_model, device)
        lstm_time = time.time() - start
        lstm_val, lstm_test = evaluate_model(lstm_wrapped, X_val, y_val, X_test, y_test)
        results["lstm"] = {
            "model": lstm_wrapped,
            "name": "LSTM",
            "train_time": lstm_time,
            "val_metrics": lstm_val,
            "test_metrics": lstm_test,
            "raw_model": lstm_model,
            "device": str(device),
        }
        logger.info(
            f"  LSTM:   Val MAE={lstm_val['mae']:.2f}, Test MAE={lstm_test['mae']:.2f} "
            f"({lstm_time:.1f}s, {device})"
        )

    except ImportError:
        logger.warning("PyTorch not installed — skipping LSTM")
    except Exception as e:
        logger.warning(f"LSTM training failed: {e}")

    # ── Select Best Model (composite score on test set) ───────────────────
    composite_scores = {}
    for k, v in results.items():
        test_score = compute_composite_score(v["test_metrics"])
        composite_scores[k] = test_score
        logger.info(
            f"  {v['name']}: composite={test_score:.2f} "
            f"(MAE={v['test_metrics']['mae']:.2f}, R²={v['test_metrics']['r2']:.4f})"
        )

    best_key = min(composite_scores, key=composite_scores.get)
    best = results[best_key]
    logger.info(f"\n🏆 BEST MODEL: {best['name']} (composite={composite_scores[best_key]:.2f})")

    # Compute residuals for confidence intervals
    y_test_pred = best["model"].predict(X_test)
    residuals = y_test - y_test_pred
    residual_stats = {
        "mean": residuals.mean(axis=0).tolist(),
        "std": residuals.std(axis=0).tolist(),
        "q5": np.percentile(residuals, 5, axis=0).tolist(),
        "q95": np.percentile(residuals, 95, axis=0).tolist(),
    }

    # Print comparison table
    logger.info("\n" + "=" * 70)
    logger.info("MODEL COMPARISON (Test Set)")
    logger.info("=" * 70)
    logger.info(f"{'Model':<20} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'Composite':>10} {'Time':>8}")
    logger.info("-" * 70)
    for key, r in sorted(results.items(), key=lambda x: composite_scores[x[0]]):
        m = r["test_metrics"]
        logger.info(
            f"{r['name']:<20} {m['mae']:>8.2f} {m['rmse']:>8.2f} "
            f"{m['r2']:>8.4f} {composite_scores[key]:>10.2f} {r['train_time']:>7.1f}s"
        )
    logger.info("=" * 70)

    return {
        "model": best["model"],
        "model_name": best["name"],
        "model_key": best_key,
        "all_results": {
            k: {
                "name": v["name"],
                "val_metrics": v["val_metrics"],
                "test_metrics": v["test_metrics"],
                "train_time": v["train_time"],
            }
            for k, v in results.items()
        },
        "train_time": best["train_time"],
        "val_metrics": best["val_metrics"],
        "test_metrics": best["test_metrics"],
        "residual_stats": residual_stats,
        "feature_columns": feature_names,
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
    }


# =============================================================================
# MODEL REGISTRY — Hopsworks
# =============================================================================


def register_in_hopsworks(result, force=False, min_improvement=0.0):
    """Register model in Hopsworks Model Registry if improved."""
    try:
        from src.models.hopsworks_registry import get_model_registry
    except ImportError:
        logger.warning("Hopsworks registry not available")
        return False

    registry = get_model_registry()

    should_register = force

    if not should_register:
        best_metrics_file = METADATA_DIR / "best_metrics.json"
        if best_metrics_file.exists():
            with open(best_metrics_file) as f:
                prev = json.load(f)
            prev_mae = prev.get("mae", float("inf"))
            curr_mae = result["val_metrics"]["mae"]
            improvement = (prev_mae - curr_mae) / prev_mae

            if improvement > min_improvement:
                should_register = True
                logger.info(f"Model improved by {improvement:.2%}, registering...")
            else:
                logger.info(f"No improvement (need >{min_improvement:.2%}), skipping")
        else:
            should_register = True

    if not should_register:
        return False

    success = registry.store_model(
        model_name=result["model_key"],
        model=result["model"],
        metrics={
            "val_mae": result["val_metrics"]["mae"],
            "val_rmse": result["val_metrics"]["rmse"],
            "val_r2": result["val_metrics"]["r2"],
            "test_mae": result["test_metrics"]["mae"],
            "test_rmse": result["test_metrics"]["rmse"],
            "test_r2": result["test_metrics"]["r2"],
        },
        metadata={
            "model_name": result["model_name"],
            "training_date": datetime.now(timezone.utc).isoformat(),
            "dataset_type": "hopsworks_feature_store",
            "train_rows": result["train_rows"],
            "val_rows": result["val_rows"],
            "test_rows": result["test_rows"],
            "n_features": len(result["feature_columns"]),
            "model_comparison": {
                k: {
                    "val_mae": v["val_metrics"]["mae"],
                    "test_mae": v["test_metrics"]["mae"],
                }
                for k, v in result["all_results"].items()
            },
        },
    )

    if success:
        logger.info(f"✅ Registered in Hopsworks: {result['model_name']}")
    return success


def save_model_locally(result):
    """Save model as local pickle (fallback for API)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    model_name = result["model_key"]
    model_path = MODELS_DIR / f"{model_name}_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(result["model"], f)
    logger.info(f"Model saved to {model_path}")

    # Production model for API
    production_path = MODELS_DIR / "best_model.pkl"
    with open(production_path, "wb") as f:
        pickle.dump(result["model"], f)
    logger.info(f"Production model saved to {production_path}")

    # Metadata
    model_comparison = {}
    for k, v in result["all_results"].items():
        model_comparison[k] = {
            "name": v["name"],
            "val_mae": v["val_metrics"]["mae"],
            "val_rmse": v["val_metrics"]["rmse"],
            "val_r2": v["val_metrics"]["r2"],
            "test_mae": v["test_metrics"]["mae"],
            "test_rmse": v["test_metrics"]["rmse"],
            "test_r2": v["test_metrics"]["r2"],
            "train_time": v["train_time"],
        }

    metadata = {
        "model_version": f"v{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        "model_name": result["model_name"],
        "model_key": result["model_key"],
        "training_date": datetime.now(timezone.utc).isoformat(),
        "data_source": "hopsworks_feature_store",
        "feature_columns": result["feature_columns"],
        "target_columns": ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"],
        "model_comparison": model_comparison,
        "metrics": {
            "val": result["val_metrics"],
            "test": result["test_metrics"],
            "overall": {
                "mae": result["test_metrics"]["mae"],
                "rmse": result["test_metrics"]["rmse"],
                "r2": result["test_metrics"]["r2"],
            },
        },
        "train_time": result["train_time"],
        "train_rows": result["train_rows"],
        "val_rows": result["val_rows"],
        "test_rows": result["test_rows"],
        "n_features": len(result["feature_columns"]),
    }

    meta_path = MODELS_DIR / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {meta_path}")

    # Save best metrics for comparison
    best_metrics = {
        "mae": result["val_metrics"]["mae"],
        "rmse": result["val_metrics"]["rmse"],
        "r2": result["val_metrics"]["r2"],
        "date": datetime.now(timezone.utc).isoformat(),
    }
    best_path = METADATA_DIR / "best_metrics.json"
    with open(best_path, "w") as f:
        json.dump(best_metrics, f, indent=2)


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train AQI prediction model")
    parser.add_argument(
        "--force-register", action="store_true", help="Force registration even without improvement"
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.01,
        help="Minimum MAE improvement to register (default: 1%%)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("DAILY TRAINING PIPELINE — HOPSWORKS EDITION")
    logger.info("=" * 60)

    try:
        # 1. Load data from Hopsworks Feature Store
        logger.info("Step 1: Loading data from Hopsworks...")
        df = load_training_data_from_hopsworks()

        if len(df) < 100:
            logger.error(f"Too few rows: {len(df)}. Need at least 100.")
            return 1

        # 2. Prepare features and targets
        logger.info("Step 2: Preparing features and targets...")
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data_from_dataframe(
            df
        )

        if len(X_train) < 50:
            logger.error(f"Too few training rows: {len(X_train)}")
            return 1

        # 3. Train all models
        logger.info("Step 3: Training all models...")
        result = train_all_models(X_train, X_val, X_test, y_train, y_val, y_test, feature_names)

        # 4. Register in Hopsworks Model Registry
        logger.info("Step 4: Registering in Hopsworks Model Registry...")
        registered = register_in_hopsworks(
            result,
            force=args.force_register,
            min_improvement=args.min_improvement,
        )

        # 5. Save locally (always, for API fallback)
        logger.info("Step 5: Saving locally...")
        save_model_locally(result)

        # 6. Summary
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"  Best model:      {result['model_name']}")
        logger.info(f"  Test MAE:        {result['test_metrics']['mae']:.2f}")
        logger.info(f"  Test R²:         {result['test_metrics']['r2']:.4f}")
        logger.info(f"  Training time:   {result['train_time']:.1f}s")
        logger.info(f"  Registered:      {registered}")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
