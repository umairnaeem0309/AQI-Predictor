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
| Model Training | ✅ VERIFIED | All 4 models on complete 4-year data |
| Model Evaluation | ✅ VERIFIED | MAE + RMSE + R² across all horizons |
| Model Selection | ✅ VERIFIED | XGBoost selected (best test metrics) |
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
| Train split (72%) | 77,034 rows | ✅ |
| Validation split (8%) | 8,559 rows | ✅ |
| Test split (20%) | 21,399 rows | ✅ |
| Duplicates | 0 | ✅ |
| Missing values | <0.2% | ✅ |

---

## Verified Model Performance (4-Year Dataset)

### Production Model: XGBoost

| Metric | Value |
|--------|-------|
| **Test MAE** | **21.34** |
| **Test RMSE** | **30.35** |
| **Test R²** | **0.6584** |
| **Train Time** | 9.9s |
| **Inference Latency** | 0.011 ms/sample |

### Overall Comparison — Test Set

| Model | MAE | RMSE | R² | Latency |
|-------|-----|------|----|---------|
| **XGBoost** | **21.34** | **30.35** | **0.6584** | 0.011 ms |
| Random Forest | 21.61 | 30.58 | 0.6533 | 0.013 ms |
| Ridge | 21.73 | 30.64 | 0.6520 | 0.0003 ms |
| LSTM | 22.95 | 32.46 | 0.6092 | 0.057 ms |

### Per-Horizon — Test Set

| Horizon | Best Model | MAE | R² |
|---------|------------|-----|----|
| 24h | XGBoost | 19.00 | 0.7206 |
| 48h | XGBoost | 21.81 | 0.6461 |
| 72h | XGBoost | 23.23 | 0.6085 |

### Why XGBoost

1. **Best Test MAE** (21.34) — lowest prediction error
2. **Best Test R²** (0.6584) — explains most variance
3. **Wins ALL 3 horizons** — 24h, 48h, 72h
4. **Fast training** (9.9s) — suitable for daily retraining
5. **Fast inference** (0.011 ms) — production-ready

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
| Registered model | XGBoost |
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
