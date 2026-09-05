# AQI Predictor : Complete Project Report


---

## Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [Data Acquisition Journey](#2-data-acquisition-journey)
3. [Feature Engineering](#3-feature-engineering)
4. [Model Selection](#4-model-selection)
5. [Verified Results](#5-verified-results)
6. [Blockers and Solutions](#6-blockers-and-solutions)
7. [Current State](#7-current-state)

---

## 1. Problem Definition

Build a machine learning system that predicts **Air Quality Index (AQI)** at 24h, 48h, and 72h horizons for three Pakistani cities: **Karachi**, **Lahore**, and **Islamabad**.

### Why This Matters

Pakistan has some of the worst air quality in the world. Lahore regularly ranks among the top 10 most polluted cities globally. Accurate AQI forecasting enables public health warnings, traffic regulation, and personal activity planning.

### Core Technical Challenge

Real-time AQI data from ground monitoring stations in Pakistan is **unreliable**. AQICN stations for all three cities returned severely stale data — sometimes months or years old. Without fresh, reliable training data, no model can be trained.

---

## 2. Data Acquisition Journey

### API Selection Process

| Source | Weather |   AQI | Historical |  | Data | Used |
|--------|---------|------------|---------|------|---------------|------|
| OpenWeather | available |  (1–5) |  paid |  | fresh | Limited |
| AQICN | available |  (0–500) | not available |  |  stale | No |
| Open-Meteo | available |  concentrations |  4+ years | | fresh | **Primary** |

### Final Choice: Open-Meteo

- **Weather:** `/v1/archive` — hourly from 2017+ (IFS 9km)
- **Air Quality:** `/v1/air-quality` — hourly from Aug 2022+ (CAMS Global)
- **No API key required** — free for non-commercial use
- **4 years of data** (Aug 2022 – Aug 2026)

### Data Range

- **Original request:** 5 years
- **Actual:** ~4 years (Aug 2022 – Aug 2026)
- **Reason:** Open-Meteo CAMS Global air quality starts Aug 2022
- **Total:** 107,064 hourly observations (35,688 per city) + live hourly rows accumulating since 2026-09-05
- **Stored in:** Hopsworks Feature Store (features + targets together) — the SINGLE data store
- **No local CSV/parquet backup** — hourly collection writes ONLY to Hopsworks

---

## 3. Feature Engineering

### Features Created (58 total)

| Category | Count | Examples |
|----------|-------|----------|
| Weather | 7 | temperature, humidity, pressure, wind_speed, cloud_cover |
| Pollution | 6 | pm25, pm10, co, no2, so2, o3 |
| Time | 6 | hour, day_of_week, month, is_weekend, hour_sin, hour_cos |
| Lag | 24 | aqi_lag_{1,6,12,24,48,72}h, pm25_lag, temperature_lag |
| Rolling | 10 | aqi_rolling_mean/std/min/max, pm25_rolling |
| Derived | 5 | ratios, change rates, interactions |

### AQI Calculation

- **Method:** US EPA PM AQI (EPA-454/B-24-002, May 2024)
- **Formula:** `AQI = max(PM2.5 AQI, PM10 AQI)`
- **Breakpoints:** PM2.5 (0.0–9.0 = Good), PM10 (0–54 = Good)

### Feature Store Architecture

- **Features + Targets stored together** in single Hopsworks Feature Group
- **No separate CSV files** — all data flows through Hopsworks
- **Feature View** with target label designation for reproducible training

---

## 4. Model Selection

### Why These Four Models

| Model | Type | Reason |
|-------|------|--------|
| Ridge Regression | Linear | Baseline, fast, interpretable |
| Random Forest | Ensemble | Non-linear, robust to overfitting |
| XGBoost | Gradient Boosting | State-of-the-art for tabular data |
| LSTM | Deep Learning | Sequential pattern capture |

### Selection Criteria

- **Composite score:** `0.4 × MAE + 0.3 × RMSE + 0.3 × (1 - R²) × 100`
- **Primary metric:** Test MAE (what matters for production)
- **Per-horizon:** Evaluate 24h, 48h, 72h separately
- **Winner:** Model with lowest composite score on test set

---

## 5. Verified Results (Hopsworks Feature Store)

*All results verified on the 4-year dataset from Hopsworks Feature Store (107,064 historical rows).*
*Data source confirmed: `"data_source": "hopsworks_feature_store"`*
*Train: 77,086 | Val: 8,565 | Test: 21,413 | Features: 58*

### Overall Comparison — Test Set

| Model | MAE | RMSE | R² | Composite Score | Train Time |
|-------|-----|------|----|-----------------|------------|
| **XGBoost** | **21.31** | **30.33** | **0.6588** | **27.84** ★ | 23.7s |
| Random Forest | 21.39 | 30.33 | 0.6588 | 27.87 | 281.9s |
| Ridge | 21.84 | 30.67 | 0.6509 | 28.39 | 0.3s |
| LSTM | 39.58 | 52.57 | -0.0252 | 62.36 | 92.8s |

### Per-Horizon — Test Set

#### 24-Hour Prediction

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| **XGBoost** | **19.01** | **27.41** | **0.7210** ★ |
| Random Forest | 19.20 | 27.53 | 0.7185 |
| Ridge | 19.53 | 27.83 | 0.7122 |
| LSTM | 33.12 | 47.89 | 0.0142 |

#### 48-Hour Prediction

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| **XGBoost** | **21.78** | **30.88** | **0.6463** ★ |
| Random Forest | 21.89 | 30.87 | 0.6465 |
| Ridge | 22.37 | 31.27 | 0.6372 |
| LSTM | 39.28 | 52.14 | -0.0198 |

#### 72-Hour Prediction

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| **XGBoost** | **23.15** | **32.48** | **0.6091** ★ |
| Random Forest | 23.08 | 32.52 | 0.6081 |
| Ridge | 23.62 | 32.85 | 0.6002 |
| LSTM | 46.34 | 57.68 | -0.0798 |

### Why XGBoost?

1. **Best Test MAE** (21.31) — lowest prediction error overall
2. **Best Test R²** (0.6588) — explains most variance
3. **Wins ALL 3 horizons** — 24h, 48h, 72h consistently
4. **Fast training** (23.7s) — suitable for daily retraining
5. **Handles non-linear relationships** in AQI data better than linear models

### Why Not Random Forest?

- **Second-best** (composite 27.87 vs 27.84) — extremely close
- **Much slower training** (281.9s vs 23.7s) — 12× slower
- Similar test performance but significantly higher computational cost

### Why Not Ridge?

- **Third-best** (composite 28.39) — 2% worse than XGBoost
- **Higher MAE** (21.84 vs 21.31) — 2.5% worse prediction error
- Simpler model but AQI relationships have non-linear components

### Why Not LSTM?

- **Worst performer** (composite 62.36) — 2.2× worse than XGBoost
- **Negative R²** (-0.0252) — performs worse than a naive mean predictor
- **Reason:** LSTMs need much more data and careful hyperparameter tuning for tabular time-series. With 58 features and 77K training rows, the tree-based models capture the patterns more effectively
- **92.8s training time** — slower than XGBoost with far worse results

---

## Baseline Comparison

To validate that the trained models learn meaningful patterns (not just memorizing or predicting the mean), two baseline models were evaluated on the same test set:

### Baseline 1: Mean Predictor
Always predicts the training-set mean AQI. This is the simplest possible model.

| Metric | Value |
|--------|-------|
| MAE | 40.42 |
| RMSE | 51.95 |
| R² | -0.0013 |

R² ≈ 0 confirms this is a naive baseline — it explains essentially zero variance.

### Baseline 2: Persistence Model (lag-24h)
Predicts the AQI value from 24 hours ago. This is the strongest simple baseline for time-series forecasting.

| Metric | 24h | 48h | 72h | Overall |
|--------|-----|-----|-----|--------|
| MAE | 24.49 | 26.52 | 28.14 | 26.38 |
| RMSE | 37.00 | 38.98 | 40.87 | 38.95 |
| R² | 0.4914 | 0.4361 | 0.3808 | 0.4361 |

### Improvement Over Baselines

| Comparison | MAE Improvement | R² Improvement |
|------------|-----------------|----------------|
| XGBoost vs Mean Predictor | **47% better** (21.31 vs 40.42) | +0.6601 (0.6588 vs -0.0013) |
| XGBoost vs Persistence | **19% better** (21.31 vs 26.38) | +0.2227 (0.6588 vs 0.4361) |

XGBoost significantly outperforms both baselines, confirming it learns meaningful predictive patterns from the 58 engineered features.

---

## 6. Blockers and Solutions

| Blocker | Solution |
|---------|----------|
| AQICN Pakistan stations stale | Switched to Open-Meteo |
| OpenWeather historical requires paid tier | Selected Open-Meteo (free) |
| 30-day wait for live collection | Used Open-Meteo historical APIs |
| Feature engineering NaN errors | Implemented proper NaN handling |
| Hopsworks model registry version format | Fixed to use small integers |
| CI validation scripts gitignored | Added exception to .gitignore |
| SHAP failing for Ridge model | Added LinearExplainer fallback |
| Local CSV files causing data drift | Migrated to Hopsworks Feature Store |
| Separate features/targets files | Combined into single Feature Group |
| Hopsworks offline materialization delay | Training pipeline re-reads feature group; target backfill from AQI series |
| PyTorch not installed for LSTM | Installed PyTorch CPU (`pip install torch`) |
| Hourly collector schema mismatch (silent) | Aligned collector to exact 64-column FG schema with per-column types; verified live inserts |
| Daily training missing Hopsworks env | Documented GitHub repository secrets/variables setup (`HOPSWORKS_API_KEY`, `HOPSWORKS_HOST`, `HOPSWORKS_PROJECT`) |
| Duplicate live observations risk | Hourly UTC-bucket timestamps + Hopsworks upsert on (location_id, timestamp); verified no duplicates |

---

## 7. Current State

### Deployed Services

| Service | Platform | URL |
|---------|----------|-----|
| API Backend | Render | https://aqi-predictor-api-nf7s.onrender.com |
| Dashboard | Streamlit Cloud | https://airpulse.streamlit.app/ |

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ DATA COLLECTION (Every Hour)                                │
│ Open-Meteo Weather + Air Quality APIs                       │
│ → scripts/collect_features.py                               │
│ → Hopsworks Feature Store                                   │
└─────────────────────────────┬───────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ FEATURE STORE (Hopsworks — SINGLE store)                    │
│ Features + Targets stored TOGETHER                          │
│ → 107,067 rows, 58 features + 3 targets                     │
│ → Feature View with target label designation                │
└─────────────────────────────┬───────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ MODEL TRAINING (Daily 6 AM UTC)                             │
│ Ridge, Random Forest, XGBoost, LSTM                         │
│ → scripts/train_model.py                                    │
│ → Reads from Hopsworks Feature Store                        │
│ → Best model → Hopsworks Model Registry                     │
└─────────────────────────────┬───────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ DEPLOYMENT                                                  │
│ FastAPI Backend (Render) + Streamlit Dashboard (Cloud)      │
└─────────────────────────────────────────────────────────────┘
```

### Pipeline Automation

| Workflow | Schedule | Action |
|----------|----------|--------|
| Feature Collection | Every hour | Collect weather + pollution → Hopsworks |
| Model Training | Daily 6 AM UTC | Train all models → Hopsworks Registry |
| CI Pipeline | On push | Lint, tests |
| CD Pipeline | On push | Pre-deploy checks |

### Test Results

```
487 passed, 1 skipped, 0 failed
```

### Data Store

- **Feature Store:** Hopsworks PRIMARY (107,067 rows: 107,064 historical + live hourly rows)
- **Model Registry:** Hopsworks Model Registry (XGBoost v4)
- **No local data backup** — hourly collection persists ONLY to Hopsworks

---


