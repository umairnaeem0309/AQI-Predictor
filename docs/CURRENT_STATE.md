# AQI Predictor — Current State

**Last Updated:** 2026-08-31  
**Status:** Production Ready — All Pipelines Verified

---

## System Overview

A production-grade AQI forecasting system that predicts Air Quality Index 24/48/72 hours ahead for Pakistani cities (Karachi, Lahore, Islamabad) using machine learning.

---

## Verification Status

| Stage | Status | Evidence |
|-------|--------|----------|
| Data Collection | ✅ VERIFIED | Open-Meteo API, 4-year range |
| Data Cleaning | ✅ VERIFIED | 107,208 rows, 0 duplicates, <0.2% NaN |
| EDA | ✅ VERIFIED | 4 Jupyter notebooks |
| Feature Engineering | ✅ VERIFIED | 63 features |
| Feature Store | ✅ VERIFIED | Hopsworks PRIMARY, 107,208 rows |
| Model Training | ✅ VERIFIED | All models on complete 4-year data |
| Model Evaluation | ✅ VERIFIED | MAE + RMSE + R² across all horizons |
| Model Selection | ✅ VERIFIED | Ridge selected (best test metrics) |
| Model Registry | ✅ VERIFIED | Hopsworks Model Registry |
| CI/CD | ✅ VERIFIED | 487 tests pass, lint clean |
| Deployment | ✅ VERIFIED | Render (API) + Streamlit Cloud (Dashboard) |

---

## Verified Data Summary

| Property | Value | Verified |
|----------|-------|----------|
| Total observations | 107,208 | ✅ |
| Cities | Karachi, Lahore, Islamabad | ✅ |
| Rows per city | 35,736 | ✅ |
| Date range | 2022-08-04 to 2026-08-28 | ✅ |
| Data coverage | ~4 years | ✅ |
| Weather features | 7 | ✅ |
| Pollution features | 6 | ✅ |
| Total features | 63 | ✅ |
| Train split | 45,370 rows | ✅ |
| Validation split | 5,042 rows | ✅ |
| Test split | 12,603 rows | ✅ |
| Duplicates | 0 | ✅ |
| Missing values | <0.2% | ✅ |

---

## Verified Model Performance (4-Year Dataset)

### Overall Comparison — Test Set

| Model | MAE | RMSE | R² | Composite | Inference Latency |
|-------|-----|------|----|-----------|-------------------|
| **Ridge** | **26.48** | **34.95** | **0.5722** | **33.91** ★ | 0.000 ms |
| Random Forest | 27.24 | 35.80 | 0.5510 | 35.11 | 0.009 ms |
| XGBoost | 28.18 | 37.26 | 0.5136 | 37.04 | 0.015 ms |

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

### Why Ridge (Updated)

1. **Best Test MAE** (26.48) — lowest prediction error overall
2. **Best Test R²** (0.5722) — explains most variance
3. **Wins ALL 3 horizons** — 24h, 48h, 72h consistently
4. **Fastest inference** (0.000 ms) — production-ready
5. **Simplest model** — most interpretable, least prone to overfitting
6. **Fast training** — suitable for daily retraining

---

## Feature Store (Hopsworks)

| Property | Value |
|----------|-------|
| Connection | ✅ eu-west.cloud.hopsworks.ai |
| Feature Group | `aqi_features_prod` v1 |
| Rows stored | 107,208 |
| Columns | 63 |
| Data source | Hopsworks Feature Store (PRIMARY) |
| Fallback | Local Parquet |

---

## Model Registry (Hopsworks)

| Property | Value |
|----------|-------|
| Platform | Hopsworks Model Registry |
| Registered model | Ridge (updated) |
| Model artifact | `models/production/best_model.pkl` |
| Comparison JSON | `models/production/model_comparison_full.json` |

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
| `feature-collection.yml` | Every hour | Collect weather + pollution |
| `daily-training.yml` | Daily 6 AM UTC | Train all models, select best |
| `ci.yml` | On push | Lint, tests |
| `ml-validation.yml` | Weekly | Data safety, feature quality |
| `cd.yml` | On push | Pre-deploy checks, Docker |

---

## Commands

```bash
# Feature collection
python scripts/collect_features.py

# Model training
python scripts/train_model.py --force-register

# Run tests
python -m pytest tests/ -v

# Start API
uvicorn app.backend.main:app --port 8000

# Start Dashboard
streamlit run app/frontend/streamlit_app.py --server.port 8501
```
