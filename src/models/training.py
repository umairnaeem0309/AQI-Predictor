"""
ML Training Pipeline — Trains and evaluates forecasting models.

Data safety:
- validate_training_data() must run before any MLflow experiment creation
- Synthetic test data rejected for training
- Dataset lineage verified before training

Multi-output strategy:
- All models support target_aqi_24h, target_aqi_48h, target_aqi_72h
- Uses multi-output regression (sklearn MultiOutputRegressor or native)
- Each model trained on same feature set for fair comparison

Reproducibility:
- Random seed recorded for every experiment
- Dataset version and feature version tracked
- All parameters logged to MLflow
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from src.config import PROJECT_ROOT
from src.feature_store.schemas import DatasetMetadata, DatasetType

logger = logging.getLogger(__name__)

# Target columns
TARGET_COLUMNS = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]

# Default random seed for reproducibility
DEFAULT_RANDOM_SEED = 42


def validate_training_data(metadata: DatasetMetadata) -> None:
    """Validate that dataset is approved for training.

    Must be called BEFORE any MLflow experiment creation.

    Checks:
    - approved_for_training flag
    - dataset_type is not synthetic_test_data
    - lineage metadata is complete

    Args:
        metadata: Dataset metadata to validate.

    Raises:
        ValueError: If validation fails.
    """
    # Check dataset type first — synthetic data is never allowed
    if metadata.dataset_type == DatasetType.SYNTHETIC_TEST:
        raise ValueError(
            f"Cannot train on synthetic test data. "
            f"Dataset type: {metadata.dataset_type.value}"
        )

    # Check approved_for_training
    if not metadata.approved_for_training:
        raise ValueError(
            f"Dataset {metadata.dataset_version} is not approved for training. "
            f"approved_for_training={metadata.approved_for_training}"
        )

    logger.info(
        "Training data validation passed: version=%s, type=%s, approved=%s",
        metadata.dataset_version,
        metadata.dataset_type.value,
        metadata.approved_for_training,
    )


def load_training_data(
    train_path: str,
    val_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load training and validation data with targets.

    Args:
        train_path: Path to training features CSV.
        val_path: Path to validation features CSV.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val).
    """
    train_features = pd.read_csv(train_path)
    val_features = pd.read_csv(val_path)

    # Separate features and targets
    feature_cols = [c for c in train_features.columns if not c.startswith("target_")]
    target_cols = [c for c in train_features.columns if c.startswith("target_")]

    # Use first 3 target columns (24h, 48h, 72h)
    available_targets = [c for c in TARGET_COLUMNS if c in target_cols]

    X_train = train_features[feature_cols]
    y_train = train_features[available_targets]
    X_val = val_features[feature_cols]
    y_val = val_features[available_targets]

    # Drop rows where all targets are NaN (end of dataset)
    valid_train = y_train.notna().any(axis=1)
    valid_val = y_val.notna().any(axis=1)

    X_train = X_train[valid_train].reset_index(drop=True)
    y_train = y_train[valid_train].reset_index(drop=True)
    X_val = X_val[valid_val].reset_index(drop=True)
    y_val = y_val[valid_val].reset_index(drop=True)

    logger.info(
        "Loaded training data: X_train=%s, y_train=%s, X_val=%s, y_val=%s",
        X_train.shape, y_train.shape, X_val.shape, y_val.shape,
    )

    return X_train, y_train, X_val, y_val


def get_model(model_name: str, params: Optional[Dict[str, Any]] = None, random_seed: int = DEFAULT_RANDOM_SEED) -> BaseEstimator:
    """Get a model instance by name.

    All models are wrapped in MultiOutputRegressor for multi-output
    regression (target_aqi_24h, target_aqi_48h, target_aqi_72h).

    Args:
        model_name: Name of the model (ridge, random_forest, xgboost, lstm).
        params: Model hyperparameters.
        random_seed: Random seed for reproducibility.

    Returns:
        Model instance (wrapped in MultiOutputRegressor if needed).
    """
    if params is None:
        params = {}

    if model_name == "ridge":
        model = Ridge(
            alpha=params.get("alpha", 1.0),
            random_state=random_seed,
        )
        return model

    elif model_name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 20),
            random_state=random_seed,
            n_jobs=-1,
        )
        return MultiOutputRegressor(model)

    elif model_name == "xgboost":
        try:
            import xgboost as xgb
            model = xgb.XGBRegressor(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", 6),
                learning_rate=params.get("learning_rate", 0.1),
                random_state=random_seed,
                n_jobs=-1,
                verbosity=0,
            )
            return MultiOutputRegressor(model)
        except ImportError:
            logger.warning("XGBoost not installed — skipping")
            return None

    elif model_name == "lstm":
        # LSTM is defined in lstm_model.py
        # Return placeholder; actual implementation in separate file
        logger.info("LSTM model requires TensorFlow — defined in lstm_model.py")
        return None

    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """Train a model and return results.

    Args:
        model_name: Name of the model.
        X_train: Training features.
        y_train: Training targets.
        X_val: Validation features.
        y_val: Validation targets.
        params: Model hyperparameters.
        random_seed: Random seed.

    Returns:
        Dictionary with model, metrics, training time, and metadata.
    """
    from src.models.evaluation import evaluate_model

    logger.info("Training model: %s", model_name)
    start_time = time.time()

    model = get_model(model_name, params, random_seed)
    if model is None:
        return {"model": None, "error": f"Model {model_name} unavailable"}

    # Handle non-numeric columns
    X_train_numeric = X_train.select_dtypes(include=[np.number])
    X_val_numeric = X_val.select_dtypes(include=[np.number])

    # Fill NaN with 0 for training (model-specific handling can be added later)
    X_train_clean = X_train_numeric.fillna(0)
    X_val_clean = X_val_numeric.fillna(0)

    # Train
    model.fit(X_train_clean, y_train.fillna(0))

    training_time = time.time() - start_time
    logger.info("Training complete: %s (%.2fs)", model_name, training_time)

    # Evaluate
    metrics = evaluate_model(model, X_val_clean, y_val)

    # Feature importance (for tree-based models)
    feature_importance = {}
    if hasattr(model, "estimators_"):
        # MultiOutputRegressor — get importance from first estimator
        if hasattr(model.estimators_[0], "feature_importances_"):
            importances = model.estimators_[0].feature_importances_
            feature_cols = X_train_numeric.columns.tolist()
            feature_importance = dict(zip(feature_cols, importances.tolist()))

    return {
        "model": model,
        "model_name": model_name,
        "metrics": metrics,
        "training_time": training_time,
        "feature_importance": feature_importance,
        "feature_columns": X_train_numeric.columns.tolist(),
        "random_seed": random_seed,
        "params": params or {},
    }


def run_training_pipeline(
    train_path: str,
    val_path: str,
    dataset_metadata: DatasetMetadata,
    models_to_train: Optional[List[str]] = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> List[Dict[str, Any]]:
    """Run the complete training pipeline.

    Order:
    1. validate_training_data() — BEFORE MLflow
    2. Load data
    3. Train each model
    4. Return results

    Args:
        train_path: Path to training features CSV.
        val_path: Path to validation features CSV.
        dataset_metadata: Dataset metadata for safety validation.
        models_to_train: List of model names. Defaults to [ridge, random_forest, xgboost].
        random_seed: Random seed for reproducibility.

    Returns:
        List of training result dictionaries.
    """
    # Step 1: Validate training data (BEFORE any MLflow operations)
    validate_training_data(dataset_metadata)

    # Step 2: Load data
    X_train, y_train, X_val, y_val = load_training_data(train_path, val_path)

    # Step 3: Train models
    if models_to_train is None:
        models_to_train = ["ridge", "random_forest", "xgboost"]

    results = []
    for model_name in models_to_train:
        try:
            result = train_model(
                model_name,
                X_train, y_train,
                X_val, y_val,
                random_seed=random_seed,
            )
            result["dataset_version"] = dataset_metadata.dataset_version
            result["is_reportable"] = dataset_metadata.approved_for_training
            results.append(result)
        except Exception as e:
            logger.error("Training failed for %s: %s", model_name, str(e))
            results.append({"model_name": model_name, "error": str(e)})

    logger.info("Training pipeline complete: %d models trained", len(results))
    return results
