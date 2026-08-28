# AQI Predictor

**Predicting Air Quality Index (AQI) for the next 3 days using a 100% serverless stack.**

A production-grade MLOps system that fetches real-time weather and pollution data, engineers features, trains ML models, and serves predictions through an interactive dashboard.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Architecture](#2-architecture)
3. [Data Collection](#3-data-collection)
4. [Feature Engineering](#4-feature-engineering)
5. [Feature Store](#5-feature-store)
6. [Model Training](#6-model-training)
7. [Model Evaluation](#7-model-evaluation)
8. [Model Registry](#8-model-registry)
9. [Deployment](#9-deployment)
10. [Automation & CI/CD](#10-automation--cicd)
11. [Monitoring](#11-monitoring)
12. [Dashboard](#12-dashboard)
13. [API Documentation](#13-api-documentation)
14. [Quick Start](#14-quick-start)
15. [Project Structure](#15-project-structure)

---

## 1. Problem Statement

**Goal:** Predict Air Quality Index (AQI) for Pakistani cities (Karachi, Lahore, Islamabad) at 24h, 48h, and 72h horizons.

**Why AQI Prediction Matters:**
- Air pollution causes 7 million premature deaths annually (WHO)
- Pakistan ranks among the most polluted countries
- Early warnings enable health precautions
- Policy makers need forecasting for intervention planning

**Target:** US EPA AQI (0-500 scale) derived from PM2.5 and PM10 using EPA NowCast methodology.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION (Hourly)                  │
│  Open-Meteo Weather + Air Quality APIs                      │
│  → scripts/collect_features.py                              │
│  → Hopsworks Feature Store (PRIMARY)                        │
│  → Local Parquet (FALLBACK)                                 │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    MODEL TRAINING (Daily)                    │
│  scripts/train_model.py                                     │
│  → Reads from Hopsworks Feature Store                       │
│  → Trains Ridge, RF, XGBoost, LSTM                          │
│  → Selects BEST model by MAE                                │
│  → Registers in MLflow Model Registry                       │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    API + DASHBOARD                           │
│  FastAPI (17 endpoints) → Streamlit (4 pages)               │
│  → Loads model from MLflow or local pickle                  │
│  → Real-time predictions for 3 cities                       │
│  → SHAP explainability, drift monitoring, alerts            │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Data Collection

### Data Source

**Open-Meteo API** — Free, no API key required, historical + forecast data.

| Endpoint | Variables | Frequency |
|----------|-----------|-----------|
| Weather API | Temperature, humidity, pressure, wind, cloud cover, precipitation | Hourly |
| Air Quality API | PM2.5, PM10, CO, NO2, SO2, O3, US AQI | Hourly |

### Historical Data

| City | Start Date | End Date | Records |
|------|------------|----------|---------|
| Karachi | 2022-08-01 | 2026-08-26 | ~35,000 |
| Lahore | 2022-08-01 | 2026-08-26 | ~35,000 |
| Islamabad | 2022-08-01 | 2026-08-26 | ~35,000 |

**Total:** ~107,000 hourly observations (4 years)

### Collection Pipeline

```bash
# Collect current data (hourly)
python scripts/collect_features.py

# Backfill historical data to Hopsworks
python scripts/backfill_hopsworks.py
```

---

## 4. Feature Engineering

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

## 5. Feature Store

### Hopsworks (PRIMARY)

Cloud feature store for versioned, reusable features.

```python
from src.feature_store import get_feature_store

store = get_feature_store()  # Connects to Hopsworks
store.insert_features("aqi_features_prod", df, metadata)
features = store.get_features("aqi_features_prod")
```

### Local Parquet (FALLBACK)

Backup when Hopsworks unavailable.

```python
from src.feature_store import get_feature_store

store = get_feature_store(prefer_local=True)
store.insert_features("aqi_features_prod", df, metadata)
```

---

## 6. Model Training

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

**Output:**
```
Model                MAE     RMSE      R²     Time
Ridge Regression   28.96    37.52  0.4762     0.2s
Random Forest      29.31    38.14  0.4493   171.7s
XGBoost            30.09    38.63  0.4365    15.3s
LSTM               30.02    38.58  0.4467   112.7s
🏆 BEST MODEL: Ridge Regression (MAE=28.96)
```

---

## 7. Model Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| **MAE** | Average prediction error magnitude |
| **RMSE** | Penalizes larger errors more heavily |
| **R²** | Proportion of variance explained |

### Final Results (XGBoost)

| Horizon | MAE | RMSE | R² |
|---------|-----|------|----|
| 24h | 19.22 | 28.36 | 0.6707 |
| 48h | 21.87 | 31.58 | 0.5887 |
| 72h | 22.87 | 32.57 | 0.5591 |
| **Overall** | **21.32** | **30.89** | **0.6065** |

---

## 8. Model Registry

### MLflow (Local)

Track every model version with metadata, metrics, and artifacts.

```python
from src.models.registry import ModelRegistry

registry = ModelRegistry()
registry.register_model(model_name, model, metrics, params, dataset_metadata, feature_columns)
registry.promote_to_production(model_name, version, dataset_type, approved, status)
```

**Features:**
- Model versioning
- Experiment comparison
- Rollback capability
- Audit trails

---

## 9. Deployment

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

---

## 10. Automation & CI/CD

### GitHub Actions

| Workflow | Schedule | What It Does |
|----------|----------|--------------|
| `feature-collection.yml` | Every hour | Collects weather + pollution, stores in Hopsworks |
| `daily-training.yml` | Daily 6 AM UTC | Trains all models, selects best, registers in MLflow |
| `ci.yml` | On push | Lint, type-check, unit tests |
| `ml-validation.yml` | Weekly | Data safety, feature quality, model artifact |
| `cd.yml` | On push | Pre-deployment, Docker build |

### Auto-Retraining

Models retrain daily on new data. If performance improves, the new model is automatically registered and deployed.

---

## 11. Monitoring

### Evidently AI

- **Data Drift Detection:** PSI-based feature distribution monitoring
- **Performance Tracking:** MAE, RMSE, R² over time
- **AQI Alerts:** Hazard detection for dangerous air quality levels

---

## 12. Dashboard

### Pages

| Page | Description |
|------|-------------|
| **Dashboard** | Live AQI predictions with confidence intervals |
| **Analytics** | Historical trends and city comparison |
| **Explainability** | Feature importance, SHAP global, SHAP per-prediction |
| **System** | Service health, monitoring, alerts |

### Running Locally

```bash
# Start API
uvicorn app.backend.main:app --port 8000

# Start Dashboard
streamlit run app/frontend/streamlit_app.py --server.port 8501
```

---

## 13. API Documentation

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/prediction` | POST | AQI prediction |
| `/model-info` | GET | Model metadata |
| `/explain/feature-importance` | GET | XGBoost feature importance |
| `/explain/shap-global` | GET | Global SHAP analysis |
| `/explain/shap-explanation` | POST | Per-prediction SHAP |
| `/monitoring/drift` | GET | Data drift detection |
| `/monitoring/performance` | GET | Training metrics |
| `/monitoring/alerts` | GET | AQI hazard alerts |
| `/monitoring/system-health` | GET | System health |
| `/data/historical` | GET | Historical data |
| `/data/statistics` | GET | City statistics |
| `/history/predictions` | GET | Prediction history |
| `/history/stats` | GET | Prediction statistics |
| `/batch/predictions` | POST | Batch predictions |

### Example Request

```bash
curl -X POST http://localhost:8000/prediction \
  -H "Content-Type: application/json" \
  -d '{"city": "Karachi"}'
```

### Example Response

```json
{
  "city": "Karachi",
  "aqi_24h": 137,
  "aqi_48h": 79,
  "aqi_72h": 138,
  "category_24h": "Unhealthy for Sensitive Groups",
  "model_version": "xgboost_v1.0.0",
  "confidence": {
    "level": 90,
    "intervals": {
      "24h": {"lower": 99, "upper": 175},
      "48h": {"lower": 55, "upper": 103},
      "72h": {"lower": 100, "upper": 176}
    }
  }
}
```

---

## 14. Quick Start

### Prerequisites

- Python 3.11
- conda or venv
- Git

### Installation

```bash
# Clone repository
git clone https://github.com/umairnaeem0309/AQI-Predictor.git
cd AQI-Predictor

# Create environment
conda create -n aqi-predictor python=3.11
conda activate aqi-predictor

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your Hopsworks credentials
```

### Generate Dataset

```bash
# Download historical data
python scripts/backfill_hopsworks.py

# Or use local data
python scripts/collect_features.py
```

### Train Models

```bash
# Train all models, select best
python scripts/train_model.py --force-register
```

### Run Application

```bash
# Start API
uvicorn app.backend.main:app --port 8000

# Start Dashboard (new terminal)
streamlit run app/frontend/streamlit_app.py --server.port 8501
```

### Run Tests

```bash
python -m pytest tests/ -v
```

---

## 15. Project Structure

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
│   └── monitoring/             # Drift detection
│
├── app/                        # Web app
│   ├── backend/                # FastAPI
│   └── frontend/               # Streamlit
│
├── notebooks/                  # EDA notebooks
├── tests/                      # Test suite
├── docs/                       # Documentation
├── data/                       # Datasets
├── models/                     # Trained models
└── .github/workflows/          # CI/CD
```

---

## License

This project is for educational purposes.

---

## Acknowledgments

- [Open-Meteo](https://open-meteo.com/) for free weather and air quality data
- [Hopsworks](https://www.hopsworks.ai/) for feature store
- [MLflow](https://mlflow.org/) for model registry
- [Streamlit](https://streamlit.io/) for dashboard
- [FastAPI](https://fastapi.tiangolo.com/) for API framework
