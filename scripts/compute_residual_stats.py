"""
Compute Residual Statistics

Loads the trained model and validation data to compute
residual statistics for confidence interval estimation.
"""

import sys
import os
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    print("=" * 60)
    print("Computing Residual Statistics for Confidence Intervals")
    print("=" * 60)

    # 1. Load model
    model_path = Path("models/production/xgboost_model.pkl")
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Loaded model: {type(model).__name__}")

    # 2. Load metadata for feature names
    meta_path = Path("models/production/model_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    feature_names = meta.get("feature_columns", [])
    target_names = meta.get("target_columns", [])
    print(f"Features: {len(feature_names)}, Targets: {len(target_names)}")

    # 3. Load validation data
    val_features_path = Path("data/processed/val_features.csv")
    val_targets_path = Path("data/processed/val_targets.csv")

    if not val_features_path.exists():
        print(f"ERROR: Validation features not found at {val_features_path}")
        return

    X_val = pd.read_csv(val_features_path)
    y_val = pd.read_csv(val_targets_path)

    # Select only model features
    available_features = [f for f in feature_names if f in X_val.columns]
    X_val = X_val[available_features].fillna(0).values

    # Select target columns
    available_targets = [t for t in target_names if t in y_val.columns]
    y_val = y_val[available_targets].fillna(0).values

    print(f"Validation samples: {len(X_val)}")

    # 4. Make predictions
    if hasattr(model, 'estimators_'):
        # MultiOutputRegressor
        predictions = np.column_stack([
            est.predict(X_val) for est in model.estimators_
        ])
    else:
        predictions = model.predict(X_val)

    print(f"Predictions shape: {predictions.shape}")
    print(f"Actuals shape: {y_val.shape}")

    # 5. Compute residuals
    from src.models.confidence import compute_residual_stats, save_residual_stats

    stats = compute_residual_stats(y_val, predictions, confidence_levels=[0.80, 0.95])

    # 6. Print summary
    print(f"\nResidual Statistics:")
    print(f"  Overall MAE: {stats['mae']:.2f}")
    print(f"  Overall RMSE: {stats['rmse']:.2f}")
    print(f"  Mean residual: {stats['mean_residual']:.2f}")
    print(f"  Std residual: {stats['std_residual']:.2f}")

    for level in [80, 95]:
        lower = stats.get(f"interval_{level}_lower", 0)
        upper = stats.get(f"interval_{level}_upper", 0)
        width = stats.get(f"interval_{level}_width", 0)
        print(f"  {level}% interval: [{lower:.1f}, {upper:.1f}] (width={width:.1f})")

    if "per_horizon" in stats:
        print(f"\nPer-Horizon Statistics:")
        for h, h_stats in stats["per_horizon"].items():
            print(f"  {h}: MAE={h_stats['mae']:.2f}, Std={h_stats['std']:.2f}")
            for level in [80, 95]:
                lower = h_stats.get(f"interval_{level}_lower", 0)
                upper = h_stats.get(f"interval_{level}_upper", 0)
                print(f"    {level}% interval: [{lower:.1f}, {upper:.1f}]")

    # 7. Save
    save_residual_stats(stats)
    print(f"\nSaved to models/production/residual_stats.json")


if __name__ == "__main__":
    main()
