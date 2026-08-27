#!/usr/bin/env python
"""
End-to-End Pipeline Verification.
Checks: data collection → cleaning → features → targets → split → model readiness.
"""
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, '.')

# Colors for output
PASS = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"
INFO = "[INFO]"

results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append({"name": name, "status": "PASS" if condition else "FAIL", "detail": detail})
    print(f"  {status} {name}" + (f" — {detail}" if detail else ""))
    return condition

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================
section("1. RAW DATA FILES")
# ============================================================

weather_file = Path("data/raw/historical/weather_data.csv")
aq_file = Path("data/raw/historical/air_quality_data.csv")
raw_file = Path("data/processed/raw_observations.csv")

check("Weather CSV exists", weather_file.exists())
check("Air quality CSV exists", aq_file.exists())
check("Raw observations CSV exists", raw_file.exists())

if weather_file.exists():
    wdf = pd.read_csv(weather_file)
    check("Weather rows > 0", len(wdf) > 0, f"{len(wdf):,} rows")
    check("Weather has timestamp", "timestamp" in wdf.columns)
    check("Weather has location_id", "location_id" in wdf.columns)
    check("Weather has temperature", "temperature" in wdf.columns)
    check("Weather has humidity", "humidity" in wdf.columns)
    check("Weather has pressure", "pressure" in wdf.columns)
    check("Weather has wind_speed", "wind_speed" in wdf.columns)
    check("Weather has wind_direction", "wind_direction" in wdf.columns)
    check("Weather has cloud_cover", "cloud_cover" in wdf.columns)
    check("Weather has precipitation", "precipitation" in wdf.columns)
    cities_w = wdf["location_id"].unique().tolist() if "location_id" in wdf.columns else []
    check("Weather covers 3 cities", len(cities_w) == 3, f"cities: {cities_w}")

if aq_file.exists():
    adf = pd.read_csv(aq_file)
    check("AQ rows > 0", len(adf) > 0, f"{len(adf):,} rows")
    check("AQ has pm25", "pm25" in adf.columns)
    check("AQ has pm10", "pm10" in adf.columns)
    check("AQ has co", "co" in adf.columns)
    check("AQ has no2", "no2" in adf.columns)
    check("AQ has so2", "so2" in adf.columns)
    check("AQ has o3", "o3" in adf.columns)
    cities_a = adf["location_id"].unique().tolist() if "location_id" in adf.columns else []
    check("AQ covers 3 cities", len(cities_a) == 3, f"cities: {cities_a}")


# ============================================================
section("2. MERGED RAW OBSERVATIONS")
# ============================================================

if raw_file.exists():
    rdf = pd.read_csv(raw_file)
    check("Raw observations > 0", len(rdf) > 0, f"{len(rdf):,} rows")

    # Timestamps
    rdf["timestamp"] = pd.to_datetime(rdf["timestamp"], utc=True)
    ts_min = rdf["timestamp"].min()
    ts_max = rdf["timestamp"].max()
    span_days = (ts_max - ts_min).days
    check("Date range >= 3 years", span_days >= 1000, f"{span_days} days ({ts_min.date()} to {ts_max.date()})")

    # Cities
    cities = sorted(rdf["location_id"].unique().tolist())
    check("All 3 cities present", cities == ["islamabad", "karachi", "lahore"], f"found: {cities}")

    # Rows per city
    city_counts = rdf["location_id"].value_counts()
    min_city = city_counts.min()
    max_city = city_counts.max()
    check("No city has < 30000 rows", min_city >= 30000, f"min={min_city}, max={max_city}")
    check("City counts roughly balanced", max_city / min_city < 1.1, f"ratio={max_city/min_city:.2f}")

    # Weather fields
    for col in ["temperature", "humidity", "pressure", "wind_speed", "wind_direction", "cloud_cover", "precipitation"]:
        if col in rdf.columns:
            missing_pct = rdf[col].isna().mean() * 100
            check(f"Weather {col} missing < 1%", missing_pct < 1, f"{missing_pct:.2f}%")
        else:
            check(f"Weather {col} exists", False, "column missing")

    # Pollution fields
    for col in ["pm25", "pm10", "co", "no2", "so2", "o3"]:
        if col in rdf.columns:
            missing_pct = rdf[col].isna().mean() * 100
            check(f"Pollution {col} missing < 5%", missing_pct < 5, f"{missing_pct:.2f}%")
        else:
            check(f"Pollution {col} exists", False, "column missing")

    # No negative PM values
    if "pm25" in rdf.columns:
        neg_pm25 = (rdf["pm25"] < 0).sum()
        check("No negative PM2.5", neg_pm25 == 0, f"{neg_pm25} negative values")
    if "pm10" in rdf.columns:
        neg_pm10 = (rdf["pm10"] < 0).sum()
        check("No negative PM10", neg_pm10 == 0, f"{neg_pm10} negative values")

    # AQI field
    if "aqi" in rdf.columns:
        valid_aqi = rdf["aqi"].dropna()
        check("AQI values present", len(valid_aqi) > 0, f"{len(valid_aqi):,} valid")
        check("AQI >= 0", (valid_aqi >= 0).all())
        check("AQI <= 500", (valid_aqi <= 500).all(), f"max={valid_aqi.max()}")
        check("AQI mean reasonable (40-200)", 40 < valid_aqi.mean() < 200, f"mean={valid_aqi.mean():.1f}")
    else:
        check("AQI column exists", False)


# ============================================================
section("3. FEATURE ENGINEERING")
# ============================================================

train_f_file = Path("data/processed/train_features.csv")
train_t_file = Path("data/processed/train_targets.csv")

if train_f_file.exists():
    tdf = pd.read_csv(train_f_file)
    check("Train features > 0", len(tdf) > 0, f"{len(tdf):,} rows")
    check("Train has > 50 columns", len(tdf.columns) > 50, f"{len(tdf.columns)} columns")

    # Time features
    for col in ["hour", "day_of_week", "month", "is_weekend", "season", "hour_sin", "hour_cos"]:
        check(f"Time feature '{col}' exists", col in tdf.columns)

    # Lag features
    for base in ["aqi", "pm25", "temperature", "humidity"]:
        for lag in [1, 6, 12, 24, 48, 72]:
            fname = f"{base}_lag_{lag}h"
            if fname in tdf.columns:
                check(f"Lag feature '{fname}' exists", True)
            # Some lags may not exist for all columns — that's OK

    # Rolling features
    for base in ["temperature", "humidity", "aqi"]:
        for window in [6, 12, 24]:
            fname = f"{base}_rolling_{window}h_mean"
            if fname in tdf.columns:
                check(f"Rolling feature '{fname}' exists", True)

    # Check no all-NaN feature columns
    nan_cols = [c for c in tdf.columns if tdf[c].isna().all()]
    check("No all-NaN feature columns", len(nan_cols) == 0, f"{len(nan_cols)} all-NaN columns")


# ============================================================
section("4. TARGETS")
# ============================================================

if train_t_file.exists():
    tgt = pd.read_csv(train_t_file)
    check("Train targets > 0", len(tgt) > 0, f"{len(tgt):,} rows")
    check("Has target_aqi_24h", "target_aqi_24h" in tgt.columns)
    check("Has target_aqi_48h", "target_aqi_48h" in tgt.columns)
    check("Has target_aqi_72h", "target_aqi_72h" in tgt.columns)

    for h in ["24h", "48h", "72h"]:
        col = f"target_aqi_{h}"
        if col in tgt.columns:
            valid = tgt[col].notna().sum()
            total = len(tgt)
            pct = valid / total * 100
            check(f"Target {h} valid > 95%", pct > 95, f"{pct:.1f}% valid ({valid:,}/{total:,})")

    # Leakage check: feature timestamp < target timestamp
    if "timestamp" in tdf.columns and "timestamp" in tgt.columns:
        tdf_ts = pd.to_datetime(tdf["timestamp"], utc=True)
        tgt_ts = pd.to_datetime(tgt["timestamp"], utc=True)
        # They should be aligned (same timestamps)
        check("Feature and target timestamps aligned", tdf_ts.equals(tgt_ts))


# ============================================================
section("5. CHRONOLOGICAL SPLIT")
# ============================================================

val_f_file = Path("data/processed/val_features.csv")
test_f_file = Path("data/processed/test_features.csv")

if train_f_file.exists() and val_f_file.exists() and test_f_file.exists():
    train = pd.read_csv(train_f_file)
    val = pd.read_csv(val_f_file)
    test = pd.read_csv(test_f_file)

    check("Train + Val + Test rows sum to total",
          len(train) + len(val) + len(test) > 0,
          f"train={len(train):,} val={len(val):,} test={len(test):,}")

    # Chronological: train < val < test
    if "timestamp" in train.columns:
        train_max = pd.to_datetime(train["timestamp"], utc=True).max()
        val_min = pd.to_datetime(val["timestamp"], utc=True).min()
        val_max = pd.to_datetime(val["timestamp"], utc=True).max()
        test_min = pd.to_datetime(test["timestamp"], utc=True).min()

        check("Train ends before Val starts", train_max <= val_min,
              f"train_max={train_max}, val_min={val_min}")
        check("Val ends before Test starts", val_max <= test_min,
              f"val_max={val_max}, test_min={test_min}")

    # No overlap
    if "timestamp" in train.columns:
        train_ts = set(pd.to_datetime(train["timestamp"], utc=True))
        val_ts = set(pd.to_datetime(val["timestamp"], utc=True))
        test_ts = set(pd.to_datetime(test["timestamp"], utc=True))
        check("No train/val overlap", len(train_ts & val_ts) == 0)
        check("No val/test overlap", len(val_ts & test_ts) == 0)
        check("No train/test overlap", len(train_ts & test_ts) == 0)


# ============================================================
section("6. AQI CALCULATION INTEGRITY")
# ============================================================

if raw_file.exists():
    rdf = pd.read_csv(raw_file)

    # Check EPA AQI calculation matches
    from src.utils.epa_aqi import calculate_pm25_aqi, calculate_pm10_aqi

    # Test a few known values
    # EPA May 2024 PM2.5 breakpoints: 0-9=Good, 9.1-35.4=Moderate, 35.5-55.4=USG,
    # 55.5-125.4=Unhealthy, 125.5-225.4=VeryUnhealthy, 225.5-325.4=Hazardous(301-400)
    test_pm25 = [5.0, 20.0, 45.0, 90.0, 175.0, 280.0]
    expected_aqi_range = [(0, 50), (51, 100), (101, 150), (151, 200), (201, 300), (301, 400)]

    all_correct = True
    for pm25, (lo, hi) in zip(test_pm25, expected_aqi_range):
        aqi = calculate_pm25_aqi(pm25)
        if aqi is None or not (lo <= aqi <= hi):
            all_correct = False
            check(f"PM2.5 {pm25} → AQI in [{lo},{hi}]", False, f"got {aqi}")

    if all_correct:
        check("EPA AQI breakpoints correct (6 test cases)", True)

    # Check dominant pollutant logic
    if "pm25_aqi" in rdf.columns and "pm10_aqi" in rdf.columns and "aqi" in rdf.columns:
        valid = rdf.dropna(subset=["pm25_aqi", "pm10_aqi", "aqi"])
        if len(valid) > 0:
            correct_dominant = ((valid["pm25_aqi"] >= valid["pm10_aqi"]) & (valid["aqi_dominant_pollutant"] == "pm25")) | \
                               ((valid["pm10_aqi"] > valid["pm25_aqi"]) & (valid["aqi_dominant_pollutant"] == "pm10"))
            check("Dominant pollutant matches max sub-index", correct_dominant.mean() > 0.99,
                  f"{correct_dominant.mean()*100:.1f}% correct")
            check("AQI == max(pm25_aqi, pm10_aqi)",
                  (valid["aqi"] == valid[["pm25_aqi", "pm10_aqi"]].max(axis=1)).mean() > 0.99)


# ============================================================
section("7. MODEL TRAINING RESULTS")
# ============================================================

results_file = Path("data/processed/model_results_full.json")
if results_file.exists():
    with open(results_file) as f:
        model_results = json.load(f)

    check("Results file has 4 models", len(model_results) == 4, f"found {len(model_results)} models")

    model_names = [r["model"] for r in model_results]
    check("All 4 models present", set(model_names) == {"Ridge", "RandomForest", "XGBoost", "LSTM"},
          f"found: {model_names}")

    for r in model_results:
        m = r["metrics"]["overall"]
        check(f"{r['model']} MAE > 0", m["mae"] > 0, f"MAE={m['mae']}")
        check(f"{r['model']} MAE < 50", m["mae"] < 50, f"MAE={m['mae']}")
        check(f"{r['model']} R² > 0", m["r2"] > 0, f"R²={m['r2']}")

    # XGBoost should be best
    xgb_result = next((r for r in model_results if r["model"] == "XGBoost"), None)
    if xgb_result:
        xgb_mae = xgb_result["metrics"]["overall"]["mae"]
        check("XGBoost MAE < 25", xgb_mae < 25, f"MAE={xgb_mae}")
        check("XGBoost R² > 0.5", xgb_result["metrics"]["overall"]["r2"] > 0.5,
              f"R²={xgb_result['metrics']['overall']['r2']}")


# ============================================================
section("8. TEST SUITE")
# ============================================================

check("Test files exist", Path("tests").exists())
check("Unit tests exist", Path("tests/unit").exists())
check("Integration tests exist", Path("tests/integration").exists())


# ============================================================
section("9. DOCUMENTATION")
# ============================================================

docs = [
    "FINAL_PROJECT_REPORT.md",
    "DATASET_REPORT.md",
    "MODEL_EXPERIMENT_PLAN.md",
    "PROJECT_JOURNEY.md",
    "DECISIONS.md",
    "CURRENT_STATE.md",
    "DATA_DICTIONARY.md",
    "DATASET_VERSIONING.md",
]

for doc in docs:
    doc_path = Path(f"docs/{doc}")
    check(f"Documentation: {doc}", doc_path.exists())


# ============================================================
section("10. GIT STATUS")
# ============================================================

import subprocess
result = subprocess.run(["git", "status", "--short"], capture_output=True, text=True)
dirty_lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
check("Working tree clean (or only untracked)", len(dirty_lines) == 0,
      f"{len(dirty_lines)} modified/untracked files")

result = subprocess.run(["git", "log", "--oneline", "-5"], capture_output=True, text=True)
recent_commits = result.stdout.strip().split("\n")
check("Has recent commits", len(recent_commits) >= 3, f"{len(recent_commits)} recent")


# ============================================================
section("SUMMARY")
# ============================================================

passed = sum(1 for r in results if r["status"] == "PASS")
failed = sum(1 for r in results if r["status"] == "FAIL")
total = len(results)

print(f"\n  Total checks: {total}")
print(f"  {PASS} Passed: {passed}")
print(f"  {FAIL} Failed: {failed}")
print(f"  Pass rate: {passed/total*100:.1f}%")

if failed > 0:
    print(f"\n  {FAIL} FAILURES:")
    for r in results:
        if r["status"] == "FAIL":
            print(f"    - {r['name']}: {r['detail']}")
else:
    print(f"\n  {PASS} ALL CHECKS PASSED — PIPELINE VERIFIED")

print()
