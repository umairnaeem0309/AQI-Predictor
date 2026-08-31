# Architecture Document

## AQI Predictor — System Architecture

**Version:** 2.0  
**Date:** 31 August 2026  
**Status:** Production Ready  

---

## 1. Overview

The AQI Predictor is a production-grade MLOps system that forecasts Air Quality Index at 24h, 48h, and 72h horizons for Pakistani cities (Karachi, Lahore, Islamabad).

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ DATA COLLECTION (Hourly)                                     │
│ Open-Meteo Weather + Air Quality APIs                        │
│ → scripts/collect_features.py                                │
└─────────────────────────────┬───────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ FEATURE ENGINEERING                                          │
│ 63 features: weather, pollution, time, lags, rolling, ratios │
│ → src/features/feature_engineering.py                        │
└─────────────────────────────┬───────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ FEATURE STORE (Hopsworks PRIMARY)                            │
│ → Hopsworks cloud (107,208 rows)                             │
│ → Local Parquet (fallback)                                   │
└─────────────────────────────┬───────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ MODEL TRAINING (Daily 6 AM UTC)                              │
│ Ridge, Random Forest, XGBoost, LSTM                          │
│ → scripts/train_model.py                                     │
└─────────────────────────────┬───────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ MODEL REGISTRY (Hopsworks)                                   │
│ Version tracking, metrics, model comparison                  │
│ → src/models/hopsworks_registry.py                           │
└─────────────────────────────┬───────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ DEPLOYMENT                                                   │
│ FastAPI Backend (Render) + Streamlit Dashboard (Cloud)       │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Data Providers

| Provider | Purpose | API Key | Status |
|----------|---------|---------|--------|
| Open-Meteo Weather | Historical + current weather | Not required | ✅ Primary |
| Open-Meteo Air Quality | Historical + current pollution | Not required | ✅ Primary |

---

## 4. Model Registry

| Component | Technology | Status |
|-----------|-----------|--------|
| Feature Store | Hopsworks (PRIMARY) | ✅ Active |
| Model Registry | Hopsworks Model Registry | ✅ Active |
| Fallback | Local Parquet + pickle | ✅ Available |

---

## 5. CI/CD Pipeline

| Workflow | Schedule | Action |
|----------|----------|--------|
| Feature Collection | Every hour | collect_features.py → Hopsworks |
| Model Training | Daily 6 AM UTC | train_model.py → Hopsworks Registry |
| CI | On push | Lint, format, tests |
| ML Validation | Weekly | Data safety, feature quality |
| CD | On push | Pre-deploy checks |

---

## 6. Deployment

| Service | Platform | URL |
|---------|----------|-----|
| API Backend | Render | https://aqi-predictor-api-nf7s.onrender.com |
| Dashboard | Streamlit Cloud | https://airpulse.streamlit.app/ |

---

## 7. Directory Structure

```
AQI-Predictor/
├── scripts/                    # Pipeline scripts
│   ├── collect_features.py     # Feature collection (hourly)
│   ├── train_model.py          # Model training (daily)
│   └── backfill_hopsworks.py   # Hopsworks backfill
│
├── src/                        # Source code
│   ├── data/                   # Data collection
│   │   └── providers/          # Open-Meteo providers
│   ├── features/               # Feature engineering
│   ├── feature_store/          # Hopsworks + Local
│   ├── models/                 # ML models
│   │   └── hopsworks_registry.py # Model registry
│   └── monitoring/             # Drift detection
│
├── app/                        # Web application
│   ├── backend/                # FastAPI (15 endpoints)
│   └── frontend/               # Streamlit (4 pages)
│
├── notebooks/                  # EDA (4 notebooks)
├── tests/                      # 487 tests
├── docs/                       # Documentation
└── .github/workflows/          # CI/CD
```

---

**Document generated:** 31 August 2026
