#!/usr/bin/env python3
"""
Daily Training Pipeline.

Reads features from the feature store, trains XGBoost,
evaluates performance, and registers in MLflow if improved.

Designed to run daily via GitHub Actions or cron.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --force-register
    python scripts/train_model.py --min-improvement 0.01
"""

import argparse
import json
import logging
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
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
FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"
MODELS_DIR = PROJECT_ROOT / "models" / "production"
METADATA_DIR = PROJECT_ROOT / "models" / "metadata"


def load_features():
    """Load features from the Feature Store.
    
    Priority:
    1. Hopsworks Feature Store (PRIMARY - cloud)
    2. Local Parquet (FALLBACK - backup)
    3. Historical dataset (CSV - last resort)
    """
    # 1. Try Hopsworks first (PRIMARY)
    try:
        from src.feature_store import get_feature_store
        
        store = get_feature_store()
        logger.info(f"Feature Store: {store.__class__.__name__}")
        
        # Try production feature group
        try:
            df = store.get_features("aqi_features_prod", version=1)
            if not df.empty:
                logger.info(f"✅ Loaded {len(df)} records from Hopsworks Feature Store")
                return df
        except Exception as e:
            logger.warning(f"Could not load from prod feature group: {e}")
        
        # Try test feature group
        try:
            df = store.get_features("aqi_features_test", version=1)
            if not df.empty:
                logger.info(f"✅ Loaded {len(df)} records from Hopsworks (test group)")
                return df
        except Exception as e:
            logger.warning(f"Could not load from test feature group: {e}")
            
    except Exception as e:
        logger.warning(f"Hopsworks connection failed: {e}")
    
    # 2. Fallback to local Parquet
    features_file = FEATURES_DIR / "hourly_observations.parquet"
    if features_file.exists():
        df = pd.read_parquet(features_file)
        logger.info(f"✅ Loaded {len(df)} records from local Parquet (fallback)")

        if "is_training_valid" in df.columns:
            valid_df = df[df["is_training_valid"] == True].copy()
            logger.info(f"Training-valid records: {len(valid_df)}")
        else:
            valid_df = df.copy()

        return valid_df

    # 3. Last resort: historical CSV
    historical_file = PROJECT_ROOT / "data" / "processed" / "train_features.csv"
    historical_targets = PROJECT_ROOT / "data" / "processed" / "train_targets.csv"

    if historical_file.exists():
        logger.info("Loading from historical dataset (CSV last resort)")
        features_df = pd.read_csv(historical_file)
        targets_df = pd.read_csv(historical_targets)

        df = pd.merge(features_df, targets_df, on=["timestamp", "location_id"], how="inner")
        
        target_cols = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
        before_drop = len(df)
        df = df.dropna(subset=target_cols)
        logger.info(f"✅ Loaded {len(df)} records from historical CSV (dropped {before_drop - len(df)} with NaN targets)")
        
        df["_targets_precomputed"] = True
        
        return df

    else:
        raise FileNotFoundError(
            f"No features found. Run collect_features.py first, "
            f"or ensure historical data exists at {historical_file}"
        )


def prepare_training_data(df):
    """
    Prepare features and targets for training.

    Uses the same feature engineering as the original pipeline.
    """
    from src.features.feature_engineering import (
        add_lag_features,
        add_rolling_features,
        add_time_features,
    )
    from src.utils.epa_aqi import calculate_pm10_aqi, calculate_pm25_aqi

    # Ensure timestamp is datetime
    if df["timestamp"].dtype == "object":
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    elif not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Sort by location and time
    df = df.sort_values(["location_id", "timestamp"]).reset_index(drop=True)

    # Add time features
    df = add_time_features(df)

    # Calculate AQI if not present
    if "aqi" not in df.columns or df["aqi"].isna().all():
        df["pm25_aqi"] = df["pm25"].apply(
            lambda x: calculate_pm25_aqi(x) if pd.notna(x) else None
        )
        df["pm10_aqi"] = df["pm10"].apply(
            lambda x: calculate_pm10_aqi(x) if pd.notna(x) else None
        )
        df["aqi"] = df[["pm25_aqi", "pm10_aqi"]].max(axis=1)

    # Add lag features
    df = add_lag_features(df)

    # Add rolling features
    df = add_rolling_features(df)

    # Add AQI-specific lags
    for lag in [1, 6, 12, 24, 48, 72]:
        df[f"aqi_lag_{lag}h"] = df.groupby("location_id")["aqi"].shift(lag)

    # Add PM lags
    for lag in [1, 24]:
        df[f"pm25_lag_{lag}h"] = df.groupby("location_id")["pm25"].shift(lag)

    # Create targets only if not already present
    if "_targets_precomputed" in df.columns and df["_targets_precomputed"].all():
        logger.info("Targets already pre-computed, skipping target generation")
        df = df.drop(columns=["_targets_precomputed"])
    else:
        for horizon, col_name in [(24, "target_aqi_24h"), (48, "target_aqi_48h"), (72, "target_aqi_72h")]:
            df[col_name] = df.groupby("location_id")["aqi"].shift(-horizon)

        # Drop rows with NaN targets
        target_cols = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
        before_drop = len(df)
        df = df.dropna(subset=target_cols)
        logger.info(f"Dropped {before_drop - len(df)} rows with missing targets, {len(df)} remaining")

    return df


def train_xgboost(df, feature_columns):
    """Train XGBoost model with multi-output regression."""
    import xgboost as xg
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.model_selection import train_test_split

    # Prepare X and y
    X = df[feature_columns].values
    y = df[["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]].values

    # Handle NaN in features
    X = np.nan_to_num(X, nan=0.0)

    # Chronological split (last 20% for test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Further split training into train/val
    val_split = int(len(X_train) * 0.9)
    X_train_final, X_val = X_train[:val_split], X_train[val_split:]
    y_train_final, y_val = y_train[:val_split], y_train[val_split:]

    logger.info(f"Train: {len(X_train_final)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train XGBoost
    start_time = time.time()

    base_model = xg.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model = MultiOutputRegressor(base_model)
    model.fit(X_train_final, y_train_final)

    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.1f}s")

    # Evaluate
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Validation metrics
    y_val_pred = model.predict(X_val)
    val_metrics = {
        "mae": float(mean_absolute_error(y_val, y_val_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_val, y_val_pred))),
        "r2": float(r2_score(y_val, y_val_pred)),
    }

    # Test metrics
    y_test_pred = model.predict(X_test)
    test_metrics = {
        "mae": float(mean_absolute_error(y_test, y_test_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
        "r2": float(r2_score(y_test, y_test_pred)),
    }

    # Per-horizon metrics
    horizon_names = ["24h", "48h", "72h"]
    for i, h in enumerate(horizon_names):
        val_metrics[f"mae_{h}"] = float(mean_absolute_error(y_val[:, i], y_val_pred[:, i]))
        val_metrics[f"rmse_{h}"] = float(np.sqrt(mean_squared_error(y_val[:, i], y_val_pred[:, i])))
        val_metrics[f"r2_{h}"] = float(r2_score(y_val[:, i], y_val_pred[:, i]))

        test_metrics[f"mae_{h}"] = float(mean_absolute_error(y_test[:, i], y_test_pred[:, i]))
        test_metrics[f"rmse_{h}"] = float(np.sqrt(mean_squared_error(y_test[:, i], y_test_pred[:, i])))
        test_metrics[f"r2_{h}"] = float(r2_score(y_test[:, i], y_test_pred[:, i]))

    # Compute residuals for confidence intervals
    residuals = y_test - y_test_pred
    residual_stats = {
        "mean": residuals.mean(axis=0).tolist(),
        "std": residuals.std(axis=0).tolist(),
        "q5": np.percentile(residuals, 5, axis=0).tolist(),
        "q95": np.percentile(residuals, 95, axis=0).tolist(),
    }

    return {
        "model": model,
        "train_time": train_time,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "residual_stats": residual_stats,
        "feature_columns": feature_columns,
        "train_rows": len(X_train_final),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
    }


def register_in_mlflow(result, force=False, min_improvement=0.0):
    """
    Register model in MLflow if performance improved.

    Returns:
        (run_id, registered) tuple
    """
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        logger.warning("MLflow not installed, skipping registration")
        return None, False

    # Check if we should register
    should_register = force

    if not should_register:
        # Load previous best metrics
        best_metrics_file = METADATA_DIR / "best_metrics.json"
        if best_metrics_file.exists():
            with open(best_metrics_file) as f:
                prev = json.load(f)

            prev_mae = prev.get("mae", float("inf"))
            curr_mae = result["val_metrics"]["mae"]

            improvement = (prev_mae - curr_mae) / prev_mae
            logger.info(f"Previous MAE: {prev_mae:.2f}, Current: {curr_mae:.2f}, Improvement: {improvement:.4f}")

            if improvement > min_improvement:
                should_register = True
                logger.info(f"Model improved by {improvement:.2%}, registering...")
            else:
                logger.info(f"Model did not improve enough (need >{min_improvement:.2%}), skipping registration")
        else:
            should_register = True
            logger.info("No previous model found, registering first model...")

    if not should_register:
        return None, False

    # Register in MLflow
    mlflow.set_experiment("aqi_predictor_production")
    run_name = f"xgboost_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}"

    with mlflow.start_run(run_name=run_name) as run:
        # Tags
        mlflow.set_tag("model_name", "xgboost_aqi_predictor")
        mlflow.set_tag("training_date", datetime.now(timezone.utc).isoformat())
        mlflow.set_tag("dataset_type", "real_api_data")
        mlflow.set_tag("approved_for_training", "true")
        mlflow.set_tag("training_pipeline", "daily_auto")

        # Parameters
        mlflow.log_param("model", "XGBoost_MultiOutput")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("train_rows", result["train_rows"])
        mlflow.log_param("val_rows", result["val_rows"])
        mlflow.log_param("test_rows", result["test_rows"])
        mlflow.log_param("n_features", len(result["feature_columns"]))

        # Metrics
        for key, value in result["val_metrics"].items():
            mlflow.log_metric(f"val_{key}", value)
        for key, value in result["test_metrics"].items():
            mlflow.log_metric(f"test_{key}", value)
        mlflow.log_metric("train_time_s", result["train_time"])

        # Model
        mlflow.sklearn.log_model(result["model"], "model")

        # Residual stats for confidence intervals
        mlflow.log_dict(result["residual_stats"], "residual_stats.json")

        # Feature list
        mlflow.log_dict({"features": result["feature_columns"]}, "feature_list.json")

        run_id = run.info.run_id
        logger.info(f"Model registered in MLflow: run_id={run_id}")

    return run_id, True


def save_model_locally(result, run_id=None):
    """Save model as local pickle (fallback for API)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = MODELS_DIR / "xgboost_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(result["model"], f)
    logger.info(f"Model saved to {model_path}")

    # Save metadata
    metadata = {
        "model_version": f"v{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        "model_name": "xgboost_aqi_predictor",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "dataset_type": "real_api_data",
        "data_provider": "open-meteo",
        "feature_version": "2.0",
        "feature_columns": result["feature_columns"],
        "target_columns": ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"],
        "metrics": {
            "val": result["val_metrics"],
            "test": result["test_metrics"],
            "overall": {
                "mae": result["test_metrics"]["mae"],
                "rmse": result["test_metrics"]["rmse"],
                "r2": result["test_metrics"]["r2"],
            },
        },
        "model_params": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
        },
        "train_time": result["train_time"],
        "train_rows": result["train_rows"],
        "val_rows": result["val_rows"],
        "test_rows": result["test_rows"],
        "n_features": len(result["feature_columns"]),
        "mlflow_run_id": run_id,
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
        "run_id": run_id,
    }
    best_path = METADATA_DIR / "best_metrics.json"
    with open(best_path, "w") as f:
        json.dump(best_metrics, f, indent=2)
    logger.info(f"Best metrics saved to {best_path}")


def main():
    parser = argparse.ArgumentParser(description="Train AQI prediction model")
    parser.add_argument("--force-register", action="store_true", help="Force registration even without improvement")
    parser.add_argument("--min-improvement", type=float, default=0.01, help="Minimum MAE improvement to register (default: 1%%)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("DAILY TRAINING PIPELINE STARTED")
    logger.info("=" * 60)

    try:
        # 1. Load features from feature store
        df = load_features()

        if len(df) < 100:
            logger.warning(f"Only {len(df)} training-valid records. Need at least 100. Skipping training.")
            return 1

        # 2. Prepare training data
        df = prepare_training_data(df)

        if len(df) < 50:
            logger.warning(f"Only {len(df)} rows after target generation. Need at least 50. Skipping.")
            return 1

        # 3. Get feature columns (exclude targets, metadata, and string columns)
        exclude_cols = [
            "timestamp", "location_id", "city_name", "data_source",
            "collected_at", "is_training_valid",
            "target_aqi_24h", "target_aqi_48h", "target_aqi_72h",
            "us_aqi", "us_aqi_pm25", "us_aqi_pm10",  # Reference AQI, not features
            "pm25_aqi", "pm10_aqi",  # Intermediate calculations
        ]
        # Also exclude string/object columns
        string_cols = df.select_dtypes(include=["object"]).columns.tolist()
        exclude_cols = list(set(exclude_cols + string_cols))
        feature_columns = [c for c in df.columns if c not in exclude_cols]
        logger.info(f"Using {len(feature_columns)} features")

        # 4. Train model
        result = train_xgboost(df, feature_columns)

        # 5. Register in MLflow
        run_id, registered = register_in_mlflow(
            result,
            force=args.force_register,
            min_improvement=args.min_improvement,
        )

        # 6. Save locally (always)
        save_model_locally(result, run_id)

        # 7. Print summary
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"  Validation MAE: {result['val_metrics']['mae']:.2f}")
        logger.info(f"  Validation R²:  {result['val_metrics']['r2']:.4f}")
        logger.info(f"  Test MAE:       {result['test_metrics']['mae']:.2f}")
        logger.info(f"  Test R²:        {result['test_metrics']['r2']:.4f}")
        logger.info(f"  Training time:  {result['train_time']:.1f}s")
        logger.info(f"  Registered:     {registered}")
        logger.info(f"  MLflow run:     {run_id}")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
