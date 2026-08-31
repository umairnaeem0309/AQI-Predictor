# AQI Predictor — Complete Project Report

**Project:** Production-grade AQI forecasting for Pakistani cities
**Timeline:** July — August 2026
**Status:** Model training complete, XGBoost selected as production model
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

---

## 3. Feature Engineering

### Features Created (63 total)

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
| Random Forest | Ensemble | Non-linear, robust |
| XGBoost | Gradient Boosting | State-of-the-art tabular |
| LSTM | Deep Learning | Sequential pattern capture |

### Selection Criteria

- **Composite score:** `0.4 × MAE + 0.3 × RMSE + 0.3 × (1 - R²) × 100`
- **Primary metric:** Test MAE (what matters for production)
- **Per-horizon:** Evaluate 24h, 48h, 72h separately

---

## 5. Verified Results (4-Year Dataset)

### Overall Comparison — Test Set

| Model | MAE | RMSE | R² | Inference Latency |
|-------|-----|------|----|-------------------|
| **XGBoost** | **21.34** | **30.35** | **0.6584** | 0.011 ms |
| Random Forest | 21.61 | 30.58 | 0.6533 | 0.013 ms |
| Ridge | 21.73 | 30.64 | 0.6520 | 0.0003 ms |
| LSTM | 22.95 | 32.46 | 0.6092 | 0.057 ms |

### Per-Horizon — Test Set

| Horizon | Best Model | MAE | RMSE | R² |
|---------|------------|-----|------|----|
| 24h | XGBoost | 19.00 | 27.43 | 0.7206 |
| 48h | XGBoost | 21.81 | 30.89 | 0.6461 |
| 72h | XGBoost | 23.23 | 32.51 | 0.6085 |

### Production Model: XGBoost

- **Test MAE:** 21.34
- **Test R²:** 0.6584
- **Wins:** All 3 horizons, overall MAE, overall R²
- **Stored in:** Hopsworks Model Registry

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
