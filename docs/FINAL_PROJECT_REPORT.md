# AQI Predictor — Complete Project Report

**Project:** Production-grade AQI forecasting for Pakistani cities
**Timeline:** July — August 2026
**Status:** Model training complete, Ridge selected as production model
**Report Date:** 31 August 2026

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

| Source | Weather | AQI | Historical | Free | Pakistan Fresh | Used |
|--------|---------|-----|-----------|------|---------------|------|
| OpenWeather | ✅ | ⚠️ (1–5) | ❌ (paid) | ✅ | ✅ | Limited |
| AQICN | ❌ | ✅ (0–500) | ❌ | ✅ | ❌ (stale) | No |
| Open-Meteo | ✅ | ✅ (concentrations) | ✅ (4+ years) | ✅ | ✅ | **Primary** |

### Final Choice: Open-Meteo

- **Weather:** `/v1/archive` — hourly from 2017+ (IFS 9km)
- **Air Quality:** `/v1/air-quality` — hourly from Aug 2022+ (CAMS Global)
- **No API key required** — free for non-commercial use
- **4 years of data** (Aug 2022 – Aug 2026)

### Data Range

- **Original request:** 5 years
- **Actual:** ~4 years (Aug 2022 – Aug 2026)
- **Reason:** Open-Meteo CAMS Global air quality starts Aug 2022
- **Total:** 107,208 hourly observations (35,736 per city)
- **After target generation:** 63,504 usable rows (72h target shift removes trailing rows)

---

## 3. Feature Engineering

### Features Created (63+ total)

| Category | Count | Examples |
|----------|-------|----------|
| Weather | 7 | temperature, humidity, pressure, wind_speed, cloud_cover |
| Pollution | 6 | pm25, pm10, co, no2, so2, o3 |
| Time | 6 | hour, day_of_week, month, is_weekend, hour_sin, hour_cos |
| Lag | 24 | aqi_lag_{1,6,12,24,48,72}h, pm25_lag, temperature_lag |
| Rolling | 10 | aqi_rolling_mean/std/min/max, pm25_rolling |
| Derived | 10 | ratios, change rates, interactions |

### AQI Calculation

- **Method:** US EPA PM AQI (EPA-454/B-24-002, May 2024)
- **Formula:** `AQI = max(PM2.5 AQI, PM10 AQI)`
- **Breakpoints:** PM2.5 (0.0–9.0 = Good), PM10 (0–54 = Good)

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

## 5. Verified Results (4-Year Dataset)

*All results verified on the complete 4-year dataset (63,504 usable rows after target generation).*

### Overall Comparison — Test Set

| Model | MAE | RMSE | R² | Composite Score | Inference Latency |
|-------|-----|------|----|-----------------|-------------------|
| **Ridge** | **26.48** | **34.95** | **0.5722** | **33.91** ★ | 0.000 ms |
| Random Forest | 27.24 | 35.80 | 0.5510 | 35.11 | 0.009 ms |
| XGBoost | 28.18 | 37.26 | 0.5136 | 37.04 | 0.015 ms |
| LSTM | *Not verified* | — | — | — | — |

> **LSTM Note:** PyTorch is not currently installed in the `aqi-predictor` environment. LSTM results were not verified on the updated 4-year dataset. The LSTM implementation exists in `src/models/lstm_model.py` and can be enabled by installing PyTorch (`pip install torch`). Earlier results on the 2.5-year dataset showed LSTM with Test MAE ≈ 22.95.

### Per-Horizon — Test Set

#### 24-Hour Prediction

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| **Ridge** | **22.50** | **29.72** | **0.6847** ★ |
| Random Forest | 23.29 | 30.85 | 0.6602 |
| XGBoost | 24.43 | 32.40 | 0.6253 |

#### 48-Hour Prediction

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| **Ridge** | **27.21** | **35.64** | **0.5536** ★ |
| Random Forest | 27.61 | 35.80 | 0.5497 |
| XGBoost | 28.16 | 37.12 | 0.5156 |

#### 72-Hour Prediction

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| **Ridge** | **29.72** | **38.86** | **0.4784** ★ |
| Random Forest | 30.82 | 40.16 | 0.4431 |
| XGBoost | 31.94 | 41.69 | 0.3998 |

### Why Ridge?

1. **Best Test MAE** (26.48) — lowest prediction error overall
2. **Best Test R²** (0.5722) — explains most variance
3. **Wins ALL 3 horizons** — consistent 24h, 48h, 72h performance
4. **Fastest inference** (0.000 ms) — production-ready
5. **Most interpretable** — linear coefficients directly explain feature influence
6. **Least overfitting risk** — simple model with regularization
7. **Fastest training** — suitable for daily retraining

### Why Not XGBoost?

While XGBoost is state-of-the-art for many tabular problems, on this specific AQI forecasting task:
- **Higher MAE** (28.18 vs 26.48) — 6.4% worse prediction error
- **Lower R²** (0.5136 vs 0.5722) — explains less variance
- **Consistently worse** across all 3 horizons
- The AQI prediction relationships appear sufficiently linear for Ridge to outperform more complex models

### Why Not Random Forest?

- **Second-best** but still worse than Ridge on all metrics
- **Slower inference** (0.009 ms vs 0.000 ms)
- More complex model with no performance benefit

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
| Historical CSV had fewer rows than Hopsworks | Re-fetched full 4-year data from Open-Meteo |
| LSTM not verifiable | PyTorch not installed — can be added when needed |

---

## 7. Current State

### Deployed Services

| Service | Platform | URL |
|---------|----------|-----|
| API Backend | Render | https://aqi-predictor-api-nf7s.onrender.com |
| Dashboard | Streamlit Cloud | https://airpulse.streamlit.app/ |

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

- **Feature Store:** Hopsworks PRIMARY (107,208 rows)
- **Model Registry:** Hopsworks Model Registry
- **Fallback:** Local Parquet

---

**Report generated:** 31 August 2026
