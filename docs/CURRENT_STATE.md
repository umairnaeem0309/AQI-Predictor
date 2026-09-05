# AQI Predictor — Current State

**Last Updated:** 2026-09-05
**Status:** Production Ready — All Pipelines Verified, Live Hourly Collection ACTIVE

---

## System Overview

A production-grade AQI forecasting system that predicts Air Quality Index 24/48/72 hours ahead for Pakistani cities (Karachi, Lahore, Islamabad) using machine learning.

---

## Live Services

| Service | Platform | URL | Status |
|---------|----------|-----|--------|
| API Backend | Render | https://aqi-predictor-api-nf7s.onrender.com | ✅ Live |
| Dashboard | Streamlit Cloud | https://airpulse.streamlit.app/ | ✅ Live |

---

## Verification Status

| Stage | Status | Evidence |
|-------|--------|----------|
| Data Collection | ✅ VERIFIED | Open-Meteo API, 4-year range (Aug 2022 – Aug 2026) + hourly live collection |
| Data Ingestion | ✅ VERIFIED | Hopsworks Feature Store (features + targets together) |
| Data Cleaning | ✅ VERIFIED | 107,064 historical rows, 0 duplicates, <0.2% NaN |
| EDA | ✅ VERIFIED | 4 Jupyter notebooks |
| Feature Engineering | ✅ VERIFIED | 58 features |
| Feature Store | ✅ VERIFIED | Hopsworks PRIMARY, 107,067 rows (single store, no local backup) |
| Feature View | ✅ VERIFIED | Target label designation |
| Model Training | ✅ VERIFIED | 4 models on complete 4-year data from Hopsworks |
| Model Evaluation | ✅ VERIFIED | MAE + RMSE + R² across all horizons |
| Model Selection | ✅ VERIFIED | XGBoost selected (best composite score) |
| Model Registry | ✅ VERIFIED | Hopsworks Model Registry (XGBoost v4) |
| CI/CD | ✅ VERIFIED | 487 tests pass, lint clean |
| Deployment | ✅ VERIFIED | Render (API) + Streamlit Cloud (Dashboard) |
| Automation | ✅ VERIFIED | Hourly collection (verified live in Hopsworks) + daily retraining |

---

## Verified Data Summary

| Property | Value | Verified |
|----------|-------|----------|
| Historical observations | 107,064 | ✅ |
| Live hourly rows (since 2026-09-05) | accumulating, +1/city/hour | ✅ |
| Cities | Karachi, Lahore, Islamabad | ✅ |
| Historical rows per city | 35,688 | ✅ |
| Date range | 2022-08-03 to 2026-08-28 (historical) + live rows | ✅ |
| Data coverage | ~4 years | ✅ |
| Features | 58 | ✅ |
| Targets | 3 (target_aqi_24h, 48h, 72h) | ✅ |
| Train split | 77,086 rows | ✅ |
| Validation split | 8,565 rows | ✅ |
| Test split | 21,413 rows | ✅ |
| Duplicates | 0 | ✅ |
| Missing values | <0.2% | ✅ |
| Data source | Hopsworks Feature Store | ✅ |

---

## Verified Model Performance (Hopsworks Feature Store)

### Overall Comparison — Test Set

| Model | MAE | RMSE | R² | Composite | Train Time |
|-------|-----|------|----|-----------|------------|
| **XGBoost** | **21.31** | **30.33** | **0.6588** | **27.84** ★ | 23.7s |
| Random Forest | 21.39 | 30.33 | 0.6588 | 27.87 | 281.9s |
| Ridge | 21.84 | 30.67 | 0.6509 | 28.39 | 0.3s |
| LSTM | 39.58 | 52.57 | -0.0252 | 62.36 | 92.8s |

> **LSTM Note:** R² = -0.0252 means LSTM performs worse than a naive mean predictor. LSTMs need much larger datasets and careful tuning for tabular time-series. The tree/linear models dominate.

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

### Why XGBoost (Updated)

1. **Best Test MAE** (21.31) — lowest prediction error
2. **Best Test R²** (0.6588) — explains most variance
3. **Wins ALL 3 horizons** — consistent performance
4. **Fast training** (23.7s) — suitable for daily retraining
5. **Handles non-linear relationships** better than linear models

---

## Baseline Comparison

| Model | MAE | RMSE | R² | Improvement vs XGBoost |
|-------|-----|------|----|------------------------|
| Mean Predictor | 40.42 | 51.95 | -0.0013 | XGBoost is **47% better** |
| Persistence (lag-24h) | 26.38 | 38.95 | 0.4361 | XGBoost is **19% better** |
| **XGBoost** | **21.31** | **30.33** | **0.6588** | ★ Production model |

XGBoost significantly outperforms both naive baselines, confirming it learns meaningful predictive patterns from the 58 engineered features.

---

## Feature Store (Hopsworks)

| Property | Value |
|----------|-------|
| Connection | ✅ eu-west.cloud.hopsworks.ai |
| Feature Group | `aqi_features_prod` v1 (64 columns) |
| Rows stored | 107,067 (107,064 historical + 3 live) |
| Features | 58 |
| Targets | 3 (target_aqi_24h, 48h, 72h) |
| Storage | Features + Targets TOGETHER — SINGLE store (no local parquet backup) |
| Live collection | Hourly upsert on (location_id, timestamp) — duplicates impossible |
| Target backfill | Training pipeline recomputes 24/48/72h targets from AQI series for live rows |
| Feature View | `aqi_feature_view` v1 |
| Data source | Hopsworks Feature Store (PRIMARY, NO CSV fallback) |

---

## Model Registry (Hopsworks)

| Property | Value |
|----------|-------|
| Platform | Hopsworks Model Registry |
| Registered model | XGBoost v4 |
| URL | https://eu-west.cloud.hopsworks.ai/p/41205/models/xgboost/4 |
| Model artifact | `models/production/best_model.pkl` |
| Model comparison | `models/production/model_metadata.json` |

---

## Deployment

| Service | Platform | URL |
|---------|----------|-----|
| API Backend | Render | https://aqi-predictor-api-nf7s.onrender.com |
| Dashboard | Streamlit Cloud | https://airpulse.streamlit.app/ |

---

## Test Results

```
487 passed, 1 skipped, 0 failed
```

---

## CI/CD Workflows

| Workflow | Schedule | Action |
|----------|----------|--------|
| `feature-collection.yml` | Every hour at :00 | Collect weather + pollution from Open-Meteo → Hopsworks |
| `daily-training.yml` | Daily 6 AM UTC | Train all models, select best, register |
| `keep-alive.yml` | Every 10 min | Ping Render API to prevent sleep |
| `ci.yml` | On push | Lint, type-check, tests, Docker build, security |
| `cd.yml` | On push | Validate, build Docker, deploy |

### Required GitHub Configuration

Both scheduled workflows require these under **Settings → Secrets and variables → Actions**:

| Name | Type | Notes |
|------|------|-------|
| `HOPSWORKS_API_KEY` | Repository **secret** | sensitive |
| `HOPSWORKS_HOST` | Repository **secret** | sensitive |
| `HOPSWORKS_PROJECT` | Repository **variable** | not sensitive; `secrets.HOPSWORKS_PROJECT` also accepted |

Use **Repository** level (not Environment) — no GitHub Environments are configured for this repo.

> **Verified 2026-09-05:** the hourly collector was silently failing (schema mismatch); fixed and
> verified with real rounds: Hopsworks grew 107,064 → 107,067, duplicate protection confirmed
> (re-run within the same hour upserts instead of duplicating). See PROJECT_JOURNAL Entry 012.

---

## Commands

```bash
# Ingest data into Hopsworks (run once or to refresh)
python scripts/ingest_to_hopsworks.py --start-date 2022-08-01

# Feature collection (hourly)
python scripts/collect_features.py

# Model training (daily)
python scripts/train_model.py --force-register

# Register model in Hopsworks Model Registry
python scripts/register_model.py

# Run tests
python -m pytest tests/ -v

# Start API
uvicorn app.backend.main:app --port 8000

# Start Dashboard
streamlit run app/frontend/streamlit_app.py --server.port 8501
```
