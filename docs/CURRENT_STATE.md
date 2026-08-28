# AQI Predictor — Current State

**Last Updated:** 2026-08-29  
**Status:** Production Ready — Auto-Retraining Enabled

---

## System Overview

A production-grade AQI forecasting system that predicts Air Quality Index 24/48/72 hours ahead for Pakistani cities using machine learning.

---

## Pipeline Status

### Feature Pipeline (Hourly)

| Component | Status | Location |
|-----------|--------|----------|
| Data Collection | ✅ Active | `scripts/collect_features.py` |
| API Provider | ✅ Open-Meteo | `src/data/providers/` |
| Feature Engineering | ✅ 71 features | `src/features/feature_engineering.py` |
| Feature Store | ✅ Hopsworks PRIMARY | `src/feature_store/hopsworks_store.py` |
| Local Fallback | ✅ Parquet | `src/feature_store/local_store.py` |

### Training Pipeline (Daily)

| Component | Status | Location |
|-----------|--------|----------|
| Data Loading | ✅ From Feature Store | `scripts/train_model.py` |
| Model Training | ✅ Ridge, RF, XGBoost, LSTM | `src/models/training.py` |
| Model Selection | ✅ Best by MAE | `src/models/selection.py` |
| Model Registry | ✅ MLflow Local | `src/models/registry.py` |
| Model Serving | ✅ MLflow → pickle | `app/services/model_service.py` |

### CI/CD Pipeline

| Workflow | Schedule | Status |
|----------|----------|--------|
| Feature Collection | Every hour | ✅ Active |
| Model Training | Daily 6 AM UTC | ✅ Active |
| CI Pipeline | On push | ✅ Active |
| ML Validation | Weekly | ✅ Active |
| CD Pipeline | On push | ✅ Active |

---

## Model Performance

| Model | MAE | RMSE | R² | Status |
|-------|-----|------|----|--------|
| Ridge Regression | 28.96 | 37.52 | 0.4762 | Tested |
| Random Forest | 29.31 | 38.14 | 0.4493 | Tested |
| **XGBoost** | **30.09** | **38.63** | **0.4365** | **Selected** |
| LSTM | 30.02 | 38.58 | 0.4467 | Tested |

---

## API Endpoints (15 total)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/prediction` | POST | AQI prediction |
| `/model-info` | GET | Model metadata |
| `/explain/feature-importance` | GET | XGBoost feature importance |
| `/explain/model-summary` | GET | Model summary |
| `/explain/shap-global` | GET | Global SHAP analysis |
| `/explain/shap-explanation` | POST | Per-prediction SHAP |
| `/monitoring/drift` | GET | Data drift detection |
| `/monitoring/performance` | GET | Training metrics |
| `/monitoring/alerts` | GET | AQI hazard alerts |
| `/monitoring/system-health` | GET | System health |
| `/data/historical` | GET | Historical data |
| `/data/statistics` | GET | City statistics |
| `/history/predictions` | GET | Prediction history |
| `/batch/predictions` | POST | Batch predictions |

---

## Dashboard Pages (4)

| Page | Description |
|------|-------------|
| **Dashboard** | Live AQI predictions with confidence intervals |
| **Analytics** | Historical trends and city comparison |
| **Explainability** | Feature importance, SHAP global, SHAP per-prediction |
| **System** | Service health, monitoring, alerts |

---

## Test Results

```
487 passed, 1 skipped, 0 failed
```

---

## Deployment

| Service | Platform | URL |
|---------|----------|-----|
| API Backend | Render | https://aqi-predictor-api.onrender.com |
| Dashboard | Streamlit Cloud | https://aqi-predictor-dashboard.streamlit.app |

Both services auto-deploy on git push to `main`.

---

## Data

| Dataset | Rows | Columns | Location |
|---------|------|---------|----------|
| Training Features | 45,722 | 71 | `data/processed/train_features.csv` |
| Validation Features | 5,081 | 71 | `data/processed/val_features.csv` |
| Test Features | 12,701 | 71 | `data/processed/test_features.csv` |
| Hopsworks Feature Store | 63,648 | 73 | Cloud |

---

## Commands

```bash
# Feature collection (hourly)
python scripts/collect_features.py

# Model training (daily)
python scripts/train_model.py --force-register

# Backfill Hopsworks
python scripts/backfill_hopsworks.py

# Run tests
python -m pytest tests/ -v

# Start API
uvicorn app.backend.main:app --port 8000

# Start Dashboard
streamlit run app/frontend/streamlit_app.py --server.port 8501
```
