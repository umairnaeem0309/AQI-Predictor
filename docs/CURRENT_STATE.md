# AQI Predictor — Current State

**Last Updated:** 2026-08-29  
**Status:** Production Ready — Auto-Retraining Enabled

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION (Hourly)                  │
│  Open-Meteo Weather API + Air Quality API                   │
│  → Feature Store (Local Parquet + Hopsworks)                │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    MODEL TRAINING (Daily)                    │
│  Feature Store → XGBoost Training → MLflow Registry         │
│  → Local Pickle (fallback)                                  │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    API + DASHBOARD                           │
│  FastAPI (17 endpoints) → Streamlit (4 pages)               │
│  Model loaded from MLflow or local pickle                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Status

| Component | Status | Location |
|-----------|--------|----------|
| **Data Collection** | ✅ Active | `scripts/collect_features.py` |
| **Feature Store** | ✅ Active | `src/feature_store/` (Local Parquet) |
| **Model Training** | ✅ Active | `scripts/train_model.py` |
| **MLflow Registry** | ✅ Active | Local MLflow tracking |
| **FastAPI Backend** | ✅ Running | `app/backend/main.py` |
| **Streamlit Dashboard** | ✅ Running | `app/frontend/streamlit_app.py` |
| **CI/CD** | ✅ Active | `.github/workflows/` |
| **Auto-Retraining** | ✅ Enabled | GitHub Actions (daily) |

---

## Automated Pipelines

### Hourly Feature Collection
- **Schedule:** Every hour via GitHub Actions
- **Workflow:** `.github/workflows/feature-collection.yml`
- **Script:** `scripts/collect_features.py`
- **What it does:**
  1. Fetches current weather + pollution from Open-Meteo
  2. Engineers features (time, lags, rolling)
  3. Calculates EPA AQI from PM2.5/PM10
  4. Stores in local Parquet feature store
  5. Deduplicates and persists

### Daily Model Training
- **Schedule:** Every day at 6 AM UTC via GitHub Actions
- **Workflow:** `.github/workflows/daily-training.yml`
- **Script:** `scripts/train_model.py`
- **What it does:**
  1. Reads features from feature store
  2. Falls back to historical dataset if insufficient data
  3. Trains XGBoost multi-output model
  4. Evaluates on validation/test sets
  5. Registers in MLflow if performance improved
  6. Saves locally as pickle (API fallback)

---

## Model Performance

| Metric | 24h | 48h | 72h | Overall |
|--------|-----|-----|-----|---------|
| **MAE** | 19.22 | 21.87 | 22.87 | 21.32 |
| **RMSE** | 28.36 | 31.58 | 32.57 | 30.89 |
| **R²** | 0.6707 | 0.5887 | 0.5591 | 0.6065 |

---

## API Endpoints (17 total)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/predict/{city}` | GET | AQI prediction |
| `/api/v1/model/info` | GET | Model metadata |
| `/api/v1/model/feature-importance` | GET | XGBoost feature importance |
| `/explain/shap-global` | GET | Global SHAP analysis |
| `/explain/shap-explanation` | POST | Per-prediction SHAP |
| `/monitoring/drift` | GET | Data drift detection |
| `/monitoring/performance` | GET | Training metrics |
| `/monitoring/alerts` | GET | AQI hazard alerts |
| `/monitoring/system-health` | GET | System health |
| `/history/predictions` | GET | Prediction history |
| `/history/stats` | GET | Prediction statistics |
| `/batch/predictions` | POST | Batch predictions |
| `/api/v1/data/raw` | GET | Raw observations |
| `/api/v1/data/processed` | GET | Processed features |
| `/api/v1/data/cities` | GET | City statistics |
| `/api/v1/data/collection-status` | GET | Collection status |

---

## Dashboard Pages (4)

1. **Dashboard** — Live AQI predictions with confidence intervals
2. **Analytics** — Historical trends and city comparison
3. **Model Explainability** — Feature importance, SHAP global, SHAP per-prediction
4. **System** — Service health, monitoring, alerts

---

## Test Results

```
648 passed, 3 skipped, 0 failed
```

---

## Deployment

| Service | Platform | URL |
|---------|----------|-----|
| **Backend (API)** | Render | `https://aqi-predictor-api.onrender.com` |
| **Dashboard** | Streamlit Cloud | `https://aqi-predictor-dashboard.streamlit.app` |

Both services auto-deploy on git push to `main`.

---

## Feature Store Architecture

### Data Flow
```
Open-Meteo API → collect_features.py → hourly_observations.parquet
                                            ↓
                              train_model.py → MLflow Registry
                                            ↓
                              API loads model → predictions
```

### Storage Locations
- **Hourly Features:** `data/processed/features/hourly_observations.parquet`
- **Collection Metadata:** `data/processed/features/collection_metadata.json`
- **Health Log:** `data/collection_health.json`
- **Historical Dataset:** `data/processed/train_features.csv` (4 years)
- **Production Model:** `models/production/xgboost_model.pkl`
- **Model Metadata:** `models/production/model_metadata.json`
- **MLflow Registry:** `mlflow.db` (local)

---

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| **XGBoost over LSTM** | Better accuracy (MAE=21.32), simpler deployment |
| **Open-Meteo over OpenWeather** | Free, no API key, historical data available |
| **US EPA PM NowCast AQI** | Standard methodology, PM2.5 + PM10 |
| **Local MLflow** | No cloud costs, sufficient for single-model setup |
| **Local Parquet** | Fast, no external dependencies for feature store |
| **GitHub Actions** | Free tier, auto-deploy on push |

---

## Remaining Work (Optional Enhancements)

| Item | Priority | Effort |
|------|----------|--------|
| Email/Slack AQI alerts | Low | Medium |
| More cities (Delhi, Mumbai) | Low | Small |
| Hopsworks cloud feature store | Low | Medium |
| Confidence intervals refinement | Low | Small |
| Mobile app | Low | Large |
