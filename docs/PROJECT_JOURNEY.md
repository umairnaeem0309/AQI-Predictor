# AQI Predictor — Project Journey Report

**Pearls AQI Predictor**  
**Predicting Air Quality Index 24/48/72 hours ahead**  
**Using a 100% serverless stack**

---

## Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [API Selection Process](#2-api-selection-process)
3. [Architecture Decisions](#3-architecture-decisions)
4. [Data Collection & Cleaning](#4-data-collection--cleaning)
5. [Feature Engineering](#5-feature-engineering)
6. [Model Selection & Experiments](#6-model-selection--experiments)
7. [Blockers & Solutions](#7-blockers--solutions)
8. [Final System Architecture](#8-final-system-architecture)
9. [Results & Evaluation](#9-results--evaluation)
10. [Deployment](#10-deployment)

---

## 1. Problem Definition

**Goal:** Predict Air Quality Index (AQI) for Pakistani cities (Karachi, Lahore, Islamabad) at 24h, 48h, and 72h horizons.

**Why AQI Prediction Matters:**
- Air pollution causes 7 million premature deaths annually (WHO)
- Pakistan ranks among the most polluted countries
- Early warnings enable health precautions
- Policy makers need forecasting for intervention planning

**Target:** US EPA AQI (0-500 scale) derived from PM2.5 and PM10 using EPA NowCast methodology.

---

## 2. API Selection Process

### Attempted Providers

| Provider | Result | Reason |
|----------|--------|--------|
| **AQICN** | ❌ Rejected | Pakistani stations months/years stale; timezone bugs; no fresh data |
| **OpenWeather** | ⚠️ Partial | Current works; historical air pollution works; historical weather requires paid plan |
| **Open-Meteo** | ✅ Selected | Free, no API key, historical + forecast data, reliable |

### Why Open-Meteo Won

1. **Free tier:** Unlimited requests (fair use)
2. **No API key required:** Simplifies deployment
3. **Historical data:** 5+ years of hourly weather + air quality
4. **Forecast data:** Up to 16 days ahead
5. **Reliable:** Consistent uptime, fast responses
6. **Comprehensive:** Temperature, humidity, wind, pressure, PM2.5, PM10, CO, NO2, SO2, O3

### Data Sources Used

| Source | Variables | Frequency |
|--------|-----------|-----------|
| Open-Meteo Weather API | Temperature, humidity, pressure, wind, cloud cover, precipitation | Hourly |
| Open-Meteo Air Quality API | PM2.5, PM10, CO, NO2, SO2, O3, US AQI | Hourly |

---

## 3. Architecture Decisions

### Decision 1: Feature Store

**Choice:** Local Parquet (primary) + Hopsworks (optional cloud)

**Rationale:**
- Local Parquet: Zero cost, fast, sufficient for single-city prediction
- Hopsworks: Available if cloud collaboration needed later
- DuckDB: SQL queries on Parquet files

### Decision 2: Model Registry

**Choice:** Local MLflow

**Rationale:**
- Free, no cloud costs
- Tracks experiments, metrics, artifacts
- Supports model versioning and promotion
- Sufficient for single-model production

### Decision 3: ML Model

**Choice:** XGBoost (Multi-Output Regressor)

**Rationale:**
- Best accuracy among tested models
- Handles tabular data well
- Fast training and inference
- Interpretable (feature importance)
- No GPU required

### Decision 4: AQI Calculation

**Choice:** US EPA PM NowCast AQI (PM2.5 + PM10)

**Rationale:**
- Standard methodology (EPA-454/B-24-002, May 2024)
- PM2.5 and PM10 are primary pollutants in Pakistan
- Preserves 0-500 scale for health communication
- Derived from Open-Meteo pollutant concentrations

### Decision 5: Deployment

**Choice:** Render (API) + Streamlit Cloud (Dashboard)

**Rationale:**
- Both free tier
- Auto-deploy on git push
- No server management
- Clean separation of concerns

---

## 4. Data Collection & Cleaning

### Historical Data Downloaded

| City | Start Date | End Date | Records |
|------|------------|----------|---------|
| Karachi | 2022-08-01 | 2026-08-26 | ~35,000 |
| Lahore | 2022-08-01 | 2026-08-26 | ~35,000 |
| Islamabad | 2022-08-01 | 2026-08-26 | ~35,000 |

**Total:** ~107,000 hourly observations (4 years)

### Data Cleaning Steps

1. **Timestamp normalization:** All timestamps converted to UTC
2. **Missing value handling:**
   - Forward-fill for gaps < 3 hours
   - Interpolation for gaps 3-24 hours
   - Drop rows with > 24 hours missing
3. **Outlier detection:**
   - PM2.5 > 500 → flagged as invalid
   - Temperature < -50 or > 60 → flagged
   - Humidity < 0 or > 100 → corrected
4. **Duplicate removal:** Deduplicated by timestamp + location
5. **AQI calculation:** EPA NowCast from PM2.5/PM10

---

## 5. Feature Engineering

### Feature Categories (71 total)

| Category | Count | Examples |
|----------|-------|----------|
| **Weather** | 7 | temperature, humidity, pressure, wind_speed, cloud_cover |
| **Pollution** | 6 | pm25, pm10, co, no2, so2, o3 |
| **Time** | 6 | hour, day_of_week, month, is_weekend, hour_sin, hour_cos |
| **AQI Lags** | 6 | aqi_lag_1h, aqi_lag_6h, aqi_lag_12h, aqi_lag_24h, aqi_lag_48h, aqi_lag_72h |
| **PM Lags** | 6 | pm25_lag_1h, pm25_lag_24h, pm10_lag_1h, pm10_lag_24h |
| **Temperature/Humidity Lags** | 6 | temperature_lag_1h, temperature_lag_24h, humidity_lag_1h, humidity_lag_24h |
| **AQI Rolling** | 6 | aqi_rolling_mean_6h, 12h, 24h, aqi_rolling_std_24h, min_24h, max_24h |
| **PM Rolling** | 4 | pm25_rolling_mean_6h, 24h, pm10_rolling_mean_24h |
| **Weather Rolling** | 2 | temperature_rolling_mean_24h, humidity_rolling_mean_24h |
| **Derived** | 12 | aqi_change_rate_1h/6h/24h, aqi_trend_24h, pm25_pm10_ratio, etc. |

### Leakage Prevention

- All lag features use `shift()` with `closed="left"` (no future data)
- Rolling windows use time-based semantics (not row-based)
- Targets are forward-shifted 24h, 48h, 72h
- Train/test split is chronological (no random shuffling)

---

## 6. Model Selection & Experiments

### Models Tested

| Model | MAE | RMSE | R² | Training Time | Inference Latency |
|-------|-----|------|----|---------------|-------------------|
| **Ridge Regression** | 28.45 | 38.21 | 0.42 | 0.8s | 0.1ms |
| **Random Forest** | 23.12 | 33.45 | 0.54 | 12.3s | 2.1ms |
| **XGBoost** | **21.32** | **30.89** | **0.61** | 20.4s | 1.8ms |
| **LSTM** | 24.56 | 35.12 | 0.49 | 180s | 5.2ms |

### Why XGBoost Was Selected

1. **Best accuracy:** Lowest MAE (21.32) and RMSE (30.89)
2. **Best R²:** 0.61 (explains 61% of variance)
3. **Fast inference:** 1.8ms per prediction (suitable for real-time)
4. **Interpretable:** Feature importance + SHAP support
5. **Robust:** Handles missing values, no scaling required
6. **Deployable:** No GPU required, small model size

### Why LSTM Was Not Selected

- Higher error than XGBoost
- 100x slower training
- Requires GPU for reasonable training time
- Harder to deploy (TensorFlow dependency)
- No significant accuracy gain

---

## 7. Blockers & Solutions

### Blocker 1: AQICN Data Staleness

**Problem:** AQICN stations for Pakistani cities returned data from March 2025 (months old)

**Solution:** Switched to Open-Meteo which provides fresh hourly data

### Blocker 2: OpenWeather Historical Weather

**Problem:** OpenWeather historical weather endpoint requires paid plan ($40/month)

**Solution:** Used Open-Meteo which provides free historical weather data

### Blocker 3: Feature Store Integration

**Problem:** Hopsworks requires API key and cloud setup

**Solution:** Implemented Local Parquet feature store as primary, Hopsworks as optional

### Blocker 4: Timezone Bugs

**Problem:** AQICN timestamps had incorrect UTC offsets (returned -05:00 for Pakistan which is UTC+5)

**Solution:** Used Open-Meteo which handles timezone correctly

### Blocker 5: Model Serving

**Problem:** FastAPI needed to load model efficiently

**Solution:** MLflow Model Registry with local pickle fallback

---

## 8. Final System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION (Hourly)                  │
│  Open-Meteo Weather + Air Quality APIs                      │
│  → scripts/collect_features.py                              │
│  → Feature Store (Local Parquet)                            │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    MODEL TRAINING (Daily)                    │
│  scripts/train_model.py                                     │
│  → Reads from Feature Store                                 │
│  → Trains XGBoost (Multi-Output)                            │
│  → Registers in MLflow                                      │
│  → Saves local pickle (fallback)                            │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    API + DASHBOARD                           │
│  FastAPI (17 endpoints) → Streamlit (4 pages)               │
│  → Loads model from MLflow or pickle                        │
│  → Real-time predictions for 3 cities                       │
│  → SHAP explainability, drift monitoring, alerts            │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. Results & Evaluation

### Model Performance (XGBoost)

| Horizon | MAE | RMSE | R² |
|---------|-----|------|----|
| 24h | 19.22 | 28.36 | 0.67 |
| 48h | 21.87 | 31.58 | 0.59 |
| 72h | 22.87 | 32.57 | 0.56 |
| **Overall** | **21.32** | **30.89** | **0.61** |

### Live Predictions (Sample)

| City | 24h AQI | Category | 48h AQI | 72h AQI |
|------|---------|----------|---------|---------|
| Karachi | 58 | Moderate | 62 | 55 |
| Lahore | 167 | Unhealthy | 155 | 148 |
| Islamabad | 159 | Unhealthy for Sensitive | 142 | 135 |

### Confidence Intervals

Predictions include 90% confidence intervals using residual quantile method:

| City | 24h AQI | 90% CI |
|------|---------|--------|
| Karachi | 58 | [35 - 81] |
| Lahore | 167 | [140 - 194] |
| Islamabad | 159 | [132 - 186] |

---

## 10. Deployment

### Services

| Service | Platform | URL |
|---------|----------|-----|
| API Backend | Render | https://aqi-predictor-api.onrender.com |
| Dashboard | Streamlit Cloud | https://aqi-predictor-dashboard.streamlit.app |

### CI/CD Pipeline

| Workflow | Trigger | What It Does |
|----------|---------|--------------|
| Feature Collection | Hourly | Fetches weather + pollution, stores in Feature Store |
| Model Training | Daily 6 AM UTC | Trains XGBoost, registers in MLflow |
| CI Pipeline | On push | Lint, type-check, unit tests, Docker build |
| ML Validation | Weekly | Data safety, feature quality, model artifact |
| CD Pipeline | On push | Pre-deployment checks, Docker build |

---

## Appendix: File Structure

```
AQI-Predictor/
├── app/                          # FastAPI + Streamlit
│   ├── backend/                  # FastAPI app
│   ├── frontend/                 # Streamlit pages
│   ├── routes/                   # API endpoints
│   ├── schemas/                  # Pydantic models
│   └── services/                 # Business logic
├── src/                          # Core modules
│   ├── config/                   # Configuration
│   ├── data/                     # Data collection
│   │   ├── providers/            # API providers
│   │   ├── live_fetcher.py       # Real-time fetcher
│   │   └── historical_ingestion.py
│   ├── features/                 # Feature engineering
│   ├── feature_store/            # Feature store
│   ├── models/                   # ML models
│   │   ├── training.py           # Training pipeline
│   │   ├── registry.py           # MLflow registry
│   │   └── confidence.py         # Confidence intervals
│   ├── monitoring/               # Drift detection
│   └── utils/                    # Utilities
├── scripts/                      # Automation
│   ├── collect_features.py       # Hourly collection
│   ├── train_model.py            # Daily training
│   └── validate_production.py    # CI validation
├── tests/                        # Test suite
├── docs/                         # Documentation
├── notebooks/                    # Jupyter notebooks
├── models/                       # Trained models
├── data/                         # Datasets
├── .github/workflows/            # CI/CD
├── Dockerfile                    # Docker config
├── render.yaml                   # Render deployment
└── requirements.txt              # Dependencies
```
