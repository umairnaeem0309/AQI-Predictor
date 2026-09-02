#!/usr/bin/env python3
"""Report training results for GitHub Actions step summary."""

import json
from pathlib import Path


def main():
    meta_path = Path("models/production/model_metadata.json")
    if not meta_path.exists():
        print("No model metadata available")
        return

    with open(meta_path) as f:
        m = json.load(f)

    test = m.get("metrics", {}).get("test", {})
    print(f"**Best Model:** {m.get('model_name', 'unknown')}")
    print(f"**Test MAE:** {test.get('mae', 0):.2f}")
    print(f"**Test RMSE:** {test.get('rmse', 0):.2f}")
    print(f"**Test R2:** {test.get('r2', 0):.4f}")
    print(f"**Data Source:** {m.get('data_source', 'unknown')}")
    print(f"**Train Rows:** {m.get('train_rows', 0):,}")
    print(f"**Features:** {m.get('n_features', 0)}")
    print(f"**Training Date:** {m.get('training_date', 'unknown')}")


if __name__ == "__main__":
    main()
