# AQI Predictor — Complete Project Journey

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
6. [Feature Store Setup](#6-feature-store-setup)
7. [Model Selection & Experiments](#7-model-selection--experiments)
8. [Model Evaluation](#8-model-evaluation)
9. [Model Registry](#9-model-registry)
10. [Deployment](#10-deployment)
11. [Automation & Monitoring](#11-automation--monitoring)
12. [Blockers & Solutions](#12-blockers--solutions)
13. [Final Results](#13-final-results)

---

## 1. Problem Definition

**Goal:** Predict Air Quality Index (AQI) for Pakistani cities (Karachi, Lahore, Islamabad) at 24h, 48h, and 72h horizons.

**Why AQI Prediction Matters:**
- Air pollution causes 7 million premature deaths annually (WHO)
- Pakistan ranks among the most polluted countries
- Early warnings enable health precautions
- Policy makers need forecasting for intervention planning

**Target:** US EPA AQI (0-500 scale) derived from PM2.5 and PM10 using EPA NowCast methodology.

**Inputs:** Weather + pollution data from external APIs  
**Outputs:** AQI predictions for 24h, 48h, 72h horizons

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

**Choice:** **Hopsworks (PRIMARY)** + Local Parquet (FALLBACK)

**Rationale:**
- Hopsworks: Centralized repository, version control, online/offline serving
- Eliminates redundant work across experiments
- Free tier available for small projects
- Local Parquet: Backup when Hopsworks unavailable

### Decision 2: Model Registry

**Choice:** MLflow (Local)

**Rationale:**
- Track every model version with metadata, metrics, artifacts
- Enable rollback, comparison, and audit trails
- Free, no cloud costs
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

## 6. Feature Store Setup

### Hopsworks Integration

```python
from src.feature_store import get_feature_store

# Connect to Hopsworks
store = get_feature_store()

# Insert features
store.insert_features("aqi_features_prod", df, metadata)

# Retrieve features
features = store.get_features("aqi_features_prod")
```

### Backfill Historical Data

```bash
# Load 4 years of data into Hopsworks
python scripts/backfill_hopsworks.py
```

**Result:** 63,648 rows loaded into Hopsworks Feature Store

---

## 7. Model Selection & Experiments

### Models Tested

| Model | Type | Rationale |
|-------|------|-----------|
| **Ridge Regression** | Linear | Simple, interpretable baseline |
| **Random Forest** | Ensemble | Handles non-linearity |
| **XGBoost** | Gradient Boosting | Best tabular performance |
| **LSTM** | Deep Learning | Sequential pattern capture |

### Training Pipeline

```bash
# Train ALL models, select best
python scripts/train_model.py --force-register
```

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

## 8. Model Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| **MAE** | Average prediction error magnitude |
| **RMSE** | Penalizes larger errors more heavily |
| **R²** | Proportion of variance explained |

### Results (All Models)

| Model | MAE | RMSE | R² | Training Time |
|-------|-----|------|----|---------------|
| **Ridge** | 28.96 | 37.52 | 0.4762 | 0.2s |
| **Random Forest** | 29.31 | 38.14 | 0.4493 | 171.7s |
| **XGBoost** | 30.09 | 38.63 | 0.4365 | 15.3s |
| **LSTM** | 30.02 | 38.58 | 0.4467 | 112.7s |

### Final XGBoost Results

| Horizon | MAE | RMSE | R² |
|---------|-----|------|----|
| 24h | 19.22 | 28.36 | 0.6707 |
| 48h | 21.87 | 31.58 | 0.5887 |
| 72h | 22.87 | 32.57 | 0.5591 |
| **Overall** | **21.32** | **30.89** | **0.6065** |

---

## 9. Model Registry

### MLflow Integration

```python
from src.models.registry import ModelRegistry

registry = ModelRegistry()

# Register model
registry.register_model(
    model_name="xgboost_v1",
    model=model,
    metrics=metrics,
    params=params,
    dataset_metadata=dataset_metadata,
    feature_columns=feature_columns
)

# Promote to production
registry.promote_to_production(
    model_name="xgboost_v1",
    version=1,
    dataset_type="real_api_data",
    approved_for_training=True,
    approval_status="approved"
)

# Load production model
model = registry.load_production_model()
```

### Version Tracking

| Version | Date | MAE | Status |
|---------|------|-----|--------|
| v1 | 2026-08-29 | 28.96 | Production |

---

## 10. Deployment

### Services

| Service | Platform | URL |
|---------|----------|-----|
| **API Backend** | Render | https://aqi-predictor-api.onrender.com |
| **Dashboard** | Streamlit Cloud | https://aqi-predictor-dashboard.streamlit.app |

### Docker

```bash
# Build
docker build -t aqi-predictor .

# Run
docker run -p 8000:8000 aqi-predictor
```

### Auto-Deploy

Both services auto-deploy on git push to `main` branch.

---

## 11. Automation & Monitoring

### GitHub Actions

| Workflow | Schedule | What It Does |
|----------|----------|--------------|
| `feature-collection.yml` | Every hour | Collects weather + pollution, stores in Hopsworks |
| `daily-training.yml` | Daily 6 AM UTC | Trains all models, selects best, registers in MLflow |
| `ci.yml` | On push | Lint, type-check, unit tests |
| `ml-validation.yml` | Weekly | Data safety, feature quality, model artifact |
| `cd.yml` | On push | Pre-deployment, Docker build |

### Monitoring

- **Data Drift Detection:** Evidently AI PSI-based monitoring
- **Performance Tracking:** MAE, RMSE, R² over time
- **AQI Alerts:** Hazard detection for dangerous air quality levels
- **Prediction History:** SQLite storage for audit trail

---

## 12. Blockers & Solutions

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

## 13. Final Results

### Live Predictions (Sample)

| City | 24h AQI | Category | 48h AQI | 72h AQI |
|------|---------|----------|---------|---------|
| Karachi | 137 | Unhealthy for Sensitive Groups | 79 | 138 |
| Lahore | 186 | Unhealthy | 166 | 141 |
| Islamabad | 131 | Unhealthy for Sensitive Groups | 123 | 137 |

### Confidence Intervals

Predictions include 90% confidence intervals using residual quantile method:

| City | 24h AQI | 90% CI |
|------|---------|--------|
| Karachi | 137 | [99 - 175] |
| Lahore | 186 | [155 - 217] |
| Islamabad | 131 | [95 - 167] |

### System Metrics

| Metric | Value |
|--------|-------|
| **Total Tests** | 487 passed |
| **API Endpoints** | 15 |
| **Dashboard Pages** | 4 |
| **Features** | 71 |
| **Historical Rows** | 107,064 |
| **Deployment** | Render + Streamlit Cloud |

---

## Appendix: Complete File Structure

```
AQI-Predictor/
├── scripts/                    # Pipeline scripts
│   ├── collect_features.py     # Feature pipeline (hourly)
│   ├── train_model.py          # Training pipeline (daily)
│   ├── backfill_hopsworks.py   # Hopsworks backfill
│   └── validate_production.py  # CI validation
│
├── src/                        # Source code
│   ├── data/                   # Data collection
│   │   ├── live_fetcher.py     # Live data
│   │   └── providers/          # API providers
│   ├── features/               # Feature engineering
│   ├── feature_store/          # Hopsworks + Local
│   ├── models/                 # ML models
│   │   ├── training.py         # Training
│   │   ├── registry.py         # MLflow registry
│   │   ├── evaluation.py       # Evaluation
│   │   └── lstm_model.py       # LSTM
│   └── monitoring/             # Drift detection
│
├── app/                        # Web app
│   ├── backend/                # FastAPI
│   │   └── routes/             # API endpoints
│   └── frontend/               # Streamlit
│       └── pages/              # Dashboard pages
│
├── notebooks/                  # EDA notebooks
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_model_experiments.ipynb
│   └── 04_model_comparison.ipynb
│
├── tests/                      # Test suite
├── docs/                       # Documentation
├── data/                       # Datasets
├── models/                     # Trained models
└── .github/workflows/          # CI/CD
```
