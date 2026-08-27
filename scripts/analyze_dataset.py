#!/usr/bin/env python
"""Generate detailed dataset statistics for the report."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

# Load raw observations
df = pd.read_csv("data/processed/raw_observations.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

print("=" * 70)
print("DATASET STATISTICS")
print("=" * 70)

# --- Overview ---
print(f"\n## Overview")
print(f"Total rows: {len(df):,}")
print(f"Cities: {df['location_id'].nunique()} — {sorted(df['location_id'].unique())}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
hours = int((df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600) + 1
print(f"Total hours spanned: {hours:,}")
print(f"Rows per city:")
for city in sorted(df["location_id"].unique()):
    c = len(df[df["location_id"] == city])
    print(f"  {city}: {c:,}")

# --- Columns ---
print(f"\n## Columns ({len(df.columns)})")
for col in sorted(df.columns):
    dtype = df[col].dtype
    missing = df[col].isna().sum()
    missing_pct = missing / len(df) * 100
    if df[col].dtype in ["float64", "int64"]:
        desc = f"min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}"
    else:
        desc = f"unique={df[col].nunique()}"
    print(f"  {col:30s} {str(dtype):10s} missing={missing:5d} ({missing_pct:5.1f}%) {desc}")

# --- AQI Distribution ---
print(f"\n## AQI Distribution")
aqi = df["aqi"].dropna()
print(f"Valid AQI values: {len(aqi):,} / {len(df):,} ({len(aqi)/len(df)*100:.1f}%)")
print(f"Range: {aqi.min():.0f} — {aqi.max():.0f}")
print(f"Mean: {aqi.mean():.1f}")
print(f"Median: {aqi.median():.1f}")
print(f"Std: {aqi.std():.1f}")

# AQI categories
from src.utils.aqi_categories import get_aqi_category
cats = {}
for v in aqi:
    try:
        _, label = get_aqi_category(int(v))
        cats[label] = cats.get(label, 0) + 1
    except:
        pass
print(f"\nAQI Categories:")
for label in ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"]:
    count = cats.get(label, 0)
    pct = count / len(aqi) * 100
    print(f"  {label:35s} {count:6d} ({pct:5.1f}%)")

# --- Per-city AQI ---
print(f"\n## Per-City AQI Statistics")
for city in sorted(df["location_id"].unique()):
    c = df[df["location_id"] == city]["aqi"].dropna()
    print(f"  {city:12s}: n={len(c):6d}, mean={c.mean():5.1f}, median={c.median():5.1f}, std={c.std():5.1f}, min={c.min():3.0f}, max={c.max():3.0f}")

# --- Missing values ---
print(f"\n## Missing Values Summary")
key_cols = ["temperature", "humidity", "pressure", "wind_speed", "pm25", "pm10", "co", "no2", "so2", "o3", "aqi"]
for col in key_cols:
    if col in df.columns:
        m = df[col].isna().sum()
        pct = m / len(df) * 100
        print(f"  {col:15s}: {m:5d} missing ({pct:5.2f}%)")

# --- Negative values ---
print(f"\n## Data Quality Issues")
for col in ["pm25", "pm10", "co", "no2", "so2", "o3"]:
    if col in df.columns:
        neg = (df[col] < 0).sum()
        if neg > 0:
            print(f"  Negative {col}: {neg} rows")
zero_aqi = (df["aqi"] == 0).sum()
print(f"  Zero AQI: {zero_aqi} rows")

# --- Feature-engineered dataset ---
print(f"\n## Feature-Engineered Dataset")
for split in ["train", "val", "test"]:
    feat_file = f"data/processed/{split}_features.csv"
    tgt_file = f"data/processed/{split}_targets.csv"
    try:
        f_df = pd.read_csv(feat_file)
        t_df = pd.read_csv(tgt_file)
        print(f"  {split}: {len(f_df):,} rows, {len(f_df.columns)} feature cols, {len(t_df.columns)} target cols")
        for tc in t_df.columns:
            if tc.startswith("target_"):
                valid = t_df[tc].notna().sum()
                print(f"    {tc}: {valid:,} valid")
    except FileNotFoundError:
        print(f"  {split}: not found")

# --- Metadata ---
print(f"\n## Metadata")
try:
    with open("data/processed/ingestion_metadata.json") as f:
        meta = json.load(f)
    for k, v in meta.items():
        if k != "rows_per_city":
            print(f"  {k}: {v}")
except:
    pass
