"""
Production Validation Script

Validates that the production model and artifacts are ready for deployment.
Used by CI/CD pipeline (GitHub Actions).
"""

import json
import sys
from pathlib import Path


def validate_model_artifacts():
    """Validate model files exist and are valid."""
    checks = []

    # Check model file
    model_path = Path("models/production/xgboost_model.pkl")
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        checks.append(("Model artifact", True, f"{size_mb:.1f} MB"))
    else:
        checks.append(("Model artifact", False, "File not found"))

    # Check metadata
    meta_path = Path("models/production/model_metadata.json")
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        metrics = meta.get("metrics", {})
        overall = metrics.get("overall", {})
        mae = overall.get("mae", 0)
        checks.append(("Model metadata", True, f"MAE={mae:.2f}"))
    else:
        checks.append(("Model metadata", False, "File not found"))

    # Check residual stats
    residual_path = Path("models/production/residual_stats.json")
    if residual_path.exists():
        checks.append(("Residual stats", True, "Available"))
    else:
        checks.append(("Residual stats", False, "Not found (optional)"))

    return checks


def validate_dataset():
    """Validate training dataset exists.

    In CI, dataset files may not be present (gitignored).
    In that case, check for the local features parquet instead.
    """
    checks = []

    train_path = Path("data/processed/train_features.csv")
    parquet_path = Path("data/processed/features/hourly_observations.parquet")

    if train_path.exists():
        import pandas as pd

        df = pd.read_csv(train_path, nrows=5)
        checks.append(("Training features", True, f"{len(df.columns)} columns (CSV)"))
    elif parquet_path.exists():
        import pandas as pd

        df = pd.read_parquet(parquet_path, columns=None)
        checks.append(("Training features", True, f"{len(df.columns)} columns (Parquet)"))
    else:
        # CI may not have data files - use model metadata to verify data was used
        meta_path = Path("models/production/model_metadata.json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            train_rows = meta.get("train_rows", 0)
            if train_rows > 0:
                checks.append(
                    (
                        "Training features",
                        True,
                        f"Verified via metadata ({train_rows} rows trained)",
                    )
                )
            else:
                checks.append(("Training features", False, "No training data evidence"))
        else:
            checks.append(("Training features", False, "File not found and no metadata"))

    val_path = Path("data/processed/val_features.csv")
    if val_path.exists():
        checks.append(("Validation features", True, "Available"))
    elif parquet_path.exists():
        checks.append(("Validation features", True, "Available (Parquet)"))
    else:
        checks.append(("Validation features", True, "Skipped (data gitignored in CI)"))

    return checks


def validate_no_synthetic_data():
    """Check no synthetic data is marked for production training."""
    checks = []

    metadata_files = list(Path("data/processed").glob("*_metadata.json"))
    for mf in metadata_files:
        try:
            with open(mf) as f:
                meta = json.load(f)
            ds_type = meta.get("dataset_type", "unknown")
            approved = meta.get("approved_for_training", False)

            if ds_type == "synthetic_test_data" and approved:
                checks.append(
                    (f"Data safety ({mf.name})", False, "Synthetic data approved for training!")
                )
            else:
                checks.append((f"Data safety ({mf.name})", True, f"type={ds_type}"))
        except Exception as e:
            checks.append((f"Data safety ({mf.name})", False, str(e)))

    if not metadata_files:
        checks.append(("Data safety", True, "No metadata files to check"))

    return checks


def main():
    print("=" * 60)
    print("Production Validation")
    print("=" * 60)

    all_checks = []
    all_checks.extend(validate_model_artifacts())
    all_checks.extend(validate_dataset())
    all_checks.extend(validate_no_synthetic_data())

    # Print results
    passed = 0
    failed = 0
    for name, success, detail in all_checks:
        status = "[OK]" if success else "[FAIL]"
        print(f"  {status} {name}: {detail}")
        if success:
            passed += 1
        else:
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

    if failed > 0:
        print("\n[FAIL] Production validation FAILED")
        sys.exit(1)
    else:
        print("\n[OK] Production validation PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
