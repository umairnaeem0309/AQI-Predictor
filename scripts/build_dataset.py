#!/usr/bin/env python
"""
Build Dataset — CLI entry point for historical data ingestion and dataset generation.

Downloads 5 years of historical weather and air quality data from Open-Meteo,
merges them, calculates EPA AQI, validates quality, and generates the ML-ready
training dataset with feature engineering and target generation.

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --start-date 2021-01-01 --end-date 2026-08-26
    python scripts/build_dataset.py --cities karachi lahore islamabad
    python scripts/build_dataset.py --skip-download  # Use previously downloaded data
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build ML-ready dataset from Open-Meteo historical data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/build_dataset.py
  python scripts/build_dataset.py --start-date 2022-08-01 --end-date 2026-08-26
  python scripts/build_dataset.py --cities karachi lahore
  python scripts/build_dataset.py --skip-download --build-only
        """,
    )
    parser.add_argument(
        "--start-date",
        default="2021-01-01",
        help="Start date for historical data (YYYY-MM-DD). Default: 2021-01-01",
    )
    parser.add_argument(
        "--end-date",
        default="2026-08-26",
        help="End date for historical data (YYYY-MM-DD). Default: 2026-08-26",
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        default=None,
        help="City IDs to include (e.g., karachi lahore islamabad). Default: all configured cities",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading; use previously downloaded data from data/raw/historical/",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only build dataset from existing merged data (implies --skip-download)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Output directory for processed datasets. Default: data/processed/",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def step1_download_ingest(args, config):
    """Step 1: Download historical data and create merged observations."""
    from src.data.historical_ingestion import run_historical_ingestion

    # Filter city configs if specific cities requested
    city_configs = config.get("cities", [])
    if args.cities:
        city_configs = [c for c in city_configs if c["id"] in args.cities]
        if not city_configs:
            print(f"ERROR: No matching cities found for: {args.cities}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print("Step 1: Historical Data Ingestion")
    print(f"{'='*60}")
    print(f"  Cities: {[c['id'] for c in city_configs]}")
    print(f"  Date range: {args.start_date} to {args.end_date}")
    print(f"  Weather: Open-Meteo Archive API (no API key required)")
    print(f"  Air Quality: Open-Meteo Air Quality API (no API key required)")
    print()

    result = run_historical_ingestion(
        city_configs=city_configs,
        start_date=args.start_date,
        end_date=args.end_date,
        save=True,
    )

    df = result["dataframe"]
    validation = result["validation"]
    metadata = result["metadata"]

    print(f"\n  Ingestion complete:")
    print(f"    Total rows: {len(df):,}")
    print(f"    Cities: {df['location_id'].nunique() if not df.empty else 0}")
    print(f"    Date range: {validation.get('date_range', {}).get('start', 'N/A')} to {validation.get('date_range', {}).get('end', 'N/A')}")
    print(f"    Validation: {validation.get('status', 'UNKNOWN')}")
    if validation.get("quality_issues"):
        print(f"    Issues: {len(validation['quality_issues'])}")
        for issue in validation["quality_issues"][:5]:
            print(f"      - {issue}")

    return df


def step2_feature_engineering(df, args):
    """Step 2: Run feature engineering pipeline."""
    from src.features.feature_engineering import engineer_features

    print(f"\n{'='*60}")
    print("Step 2: Feature Engineering")
    print(f"{'='*60}")

    if df.empty:
        print("  WARNING: Empty DataFrame — skipping feature engineering")
        return df

    df_features = engineer_features(df)
    print(f"  Features generated: {len(df_features.columns)} columns")
    print(f"  Rows: {len(df_features):,}")

    return df_features


def step3_target_generation(df, args):
    """Step 3: Generate 24h/48h/72h forecast targets."""
    from src.data.dataset_builder import generate_targets

    print(f"\n{'='*60}")
    print("Step 3: Target Generation")
    print(f"{'='*60}")

    if df.empty:
        print("  WARNING: Empty DataFrame — skipping target generation")
        return df

    df_targets = generate_targets(df, horizons=[24, 48, 72])

    for horizon in [24, 48, 72]:
        col = f"target_aqi_{horizon}h"
        if col in df_targets.columns:
            valid = df_targets[col].notna().sum()
            print(f"  target_aqi_{horizon}h: {valid:,} valid rows out of {len(df_targets):,}")

    return df_targets


def step4_chronological_split(df, args):
    """Step 4: Split data chronologically into train/val/test."""
    from src.data.dataset_builder import split_chronological

    print(f"\n{'='*60}")
    print("Step 4: Chronological Train/Val/Test Split")
    print(f"{'='*60}")

    if df.empty:
        print("  WARNING: Empty DataFrame — skipping split")
        return {}, {}

    # Use year-based split for time series
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True)
        year = ts.dt.year

        # Training: earliest years, Validation: middle, Test: latest
        years_sorted = sorted(year.unique())
        if len(years_sorted) >= 3:
            # Split: ~70% train, ~15% val, ~15% test by time
            n_years = len(years_sorted)
            train_end_idx = int(n_years * 0.7)
            val_end_idx = int(n_years * 0.85)

            train_years = years_sorted[:train_end_idx]
            val_years = years_sorted[train_end_idx:val_end_idx]
            test_years = years_sorted[val_end_idx:]

            train_df = df[year.isin(train_years)].copy()
            val_df = df[year.isin(val_years)].copy()
            test_df = df[year.isin(test_years)].copy()

            print(f"  Train: {train_years[0]}-{train_years[-1]} ({len(train_df):,} rows)")
            print(f"  Val:   {val_years[0]}-{val_years[-1]} ({len(val_df):,} rows)")
            print(f"  Test:  {test_years[0]}-{test_years[-1]} ({len(test_df):,} rows)")

            return {"train": train_df, "val": val_df, "test": test_df}, {}

    # Fallback: percentage-based chronological split
    train_df, val_df, test_df = split_chronological(df)
    print(f"  Train: {len(train_df):,} rows")
    print(f"  Val:   {len(val_df):,} rows")
    print(f"  Test:  {len(test_df):,} rows")

    return {"train": train_df, "val": val_df, "test": test_df}, {}


def step5_save_datasets(splits, args):
    """Step 5: Save train/val/test datasets to disk."""
    import pandas as pd

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Step 5: Save Datasets")
    print(f"{'='*60}")

    for split_name, split_df in splits.items():
        if split_df is None or (isinstance(split_df, pd.DataFrame) and split_df.empty):
            print(f"  {split_name}: empty — skipping")
            continue

        # Save features (all columns except targets)
        target_cols = [c for c in split_df.columns if c.startswith("target_")]
        feature_cols = [c for c in split_df.columns if c not in target_cols]

        features_file = output_dir / f"{split_name}_features.csv"
        targets_file = output_dir / f"{split_name}_targets.csv"

        split_df[feature_cols].to_csv(features_file, index=False)
        split_df[target_cols + ["timestamp", "location_id"]].to_csv(
            targets_file, index=False
        )

        print(f"  {split_name}: {len(split_df):,} rows → {features_file.name}, {targets_file.name}")

    # Save metadata
    metadata = {
        "generation_timestamp": __import__("datetime").datetime.utcnow().isoformat(),
        "splits": {
            name: len(df) if df is not None and not (isinstance(df, pd.DataFrame) and df.empty) else 0
            for name, df in splits.items()
        },
        "dataset_type": "real_api_data",
        "approved_for_training": True,
        "data_provider": "open-meteo",
    }
    metadata_file = output_dir / "dataset_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Metadata: {metadata_file.name}")


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load config
    config = load_config()

    print("=" * 60)
    print("AQI Predictor — Historical Dataset Builder")
    print("=" * 60)
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Start: {args.start_date}")
    print(f"  End: {args.end_date}")
    print(f"  Output: {args.output_dir}")
    print()

    start_time = time.time()

    if args.build_only:
        args.skip_download = True

    # Step 1: Download and ingest
    if args.skip_download:
        # Load previously merged data
        merged_file = args.output_dir.parent / "raw" / "historical" / "merged_observations.csv"
        if not merged_file.exists():
            merged_file = args.output_dir / "raw_observations.csv"
        if merged_file.exists():
            import pandas as pd
            print(f"Loading previously merged data from {merged_file}")
            df = pd.read_csv(merged_file)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        else:
            print(f"ERROR: No previously downloaded data found at {merged_file}")
            print("Run without --skip-download first.")
            sys.exit(1)
    else:
        df = step1_download_ingest(args, config)

    # Step 2: Feature engineering
    if not args.skip_download or args.build_only:
        df = step2_feature_engineering(df, args)

        # Step 3: Target generation
        df = step3_target_generation(df, args)

        # Step 4: Chronological split
        splits, _ = step4_chronological_split(df, args)

        # Step 5: Save
        step5_save_datasets(splits, args)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Complete! Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}")


import pandas as pd

if __name__ == "__main__":
    main()
