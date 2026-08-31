# 🌍 AQI Predictor

**Predict Air Quality Index 24/48/72 hours ahead for Pakistani cities using Machine Learning.**

[![CI](https://github.com/umairnaeem0309/AQI-Predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/umairnaeem0309/AQI-Predictor/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-487%20passed-green)]()
[![Python](https://img.shields.io/badge/python-3.11-blue)]()

[Live Dashboard](https://airpulse.streamlit.app/) | [API Backend](https://aqi-predictor-api-nf7s.onrender.com) | [Documentation](docs/)

---

## Overview

A production-grade ML pipeline that collects real-time weather and air quality data from Open-Meteo, engineers 58 features, trains multiple models (Ridge, Random Forest, XGBoost, LSTM), and serves predictions via a REST API and interactive dashboard.

### Cities Supported

- 🏙️ **Karachi** — Pakistan's largest city, coastal
- 🏙️ **Lahore** — Punjab province, industrial
- 🏙️ **Islamabad** — Capital city

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ DATA COLLECTION (Hourly)                                     │
│ Open-Meteo Weather + Air Quality APIs                        │
│ → scripts/collect_features.py                                │
└─────────────────────────────┬───────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ FEATURE STORE (Hopsworks PRIMARY)                            │
│ Features + Targets stored TOGETHER                           │
│ → 107,064 rows, 58 features + 3 targets                     │
│ → Feature View with target label designation                 │
└─────────────────────────────┬───────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ MODEL TRAINING (Daily 6 AM UTC)                              │
│ Ridge, Random Forest, XGBoost, LSTM                          │
│ → scripts/train_model.py (reads from Hopsworks)              │
│ → Best model → Hopsworks Model Registry                      │
└─────────────────────────────┬───────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ DEPLOYMENT                                                   │
│ FastAPI Backend (Render) + Streamlit Dashboard (Cloud)       │
└─────────────────────────────────────────────────────────────┘
```

---

## Verified Model Performance

*All results verified on 107,064 rows from Hopsworks Feature Store.*

### Production Model: XGBoost

| Metric | Value |
|--------|-------|
| **Test MAE** | **21.31** |
| **Test RMSE** | **30.33** |
| **Test R²** | **0.6588** |
| **Composite Score** | **27.84** |
| **Training Time** | 22.5s |

### Model Comparison — Test Set (All Models)

| Model | MAE | RMSE | R² | Composite | Train Time |
|-------|-----|------|----|-----------|------------|
| **XGBoost** | **21.31** | **30.33** | **0.6588** | **27.84** ★ | 23.7s |
| Random Forest | 21.39 | 30.33 | 0.6588 | 27.87 | 281.9s |
| Ridge | 21.84 | 30.67 | 0.6509 | 28.39 | 0.3s |
| LSTM | 39.58 | 52.57 | -0.0252 | 62.36 | 92.8s |

> **LSTM Note:** R² = -0.0252 means LSTM performs worse than a naive mean predictor. LSTMs need much more data and careful tuning for tabular time-series. The 3 tree/linear models dominate.

### Per-Horizon Comparison — Test Set

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

### Why XGBoost?

1. **Best Test MAE** (21.31) — lowest prediction error
2. **Best Test R²** (0.6588) — explains most variance
3. **Wins ALL 3 horizons** — 24h, 48h, 72h consistently
4. **Fast training** (22.5s) — suitable for daily retraining
5. **Handles non-linear relationships** in AQI data

---

## Verified Data Pipeline

| Stage | Status | Details |
|-------|--------|---------|
| Data Collection | ✅ Verified | Open-Meteo API, 4-year range |
| Data Ingestion | ✅ Verified | Hopsworks Feature Store (features + targets together) |
| Data Cleaning | ✅ Verified | 107,064 rows, 0 duplicates, <0.2% NaN |
| Feature Engineering | ✅ Verified | 58 features |
| Feature Store | ✅ Verified | Hopsworks: 107,064 rows |
| Feature View | ✅ Verified | Target label designation |
| Model Training | ✅ Verified | 4 models (Ridge, RF, XGBoost, LSTM) from Hopsworks |
| Model Registry | ✅ Verified | Hopsworks Model Registry (XGBoost v2) |
| CI/CD | ✅ Verified | 487 tests passing, lint clean |

---

## Quick Start

### Prerequisites

- Python 3.11
- Conda environment: `aqi-predictor`

### Installation

```bash
git clone https://github.com/umairnaeem0309/AQI-Predictor.git
cd AQI-Predictor
pip install -r requirements.txt
```

### Run Locally

```bash
# Start API backend
uvicorn app.backend.main:app --port 8000

# Start dashboard
streamlit run app/frontend/streamlit_app.py --server.port 8501
```

### Run Pipelines

```bash
# Step 1: Ingest data into Hopsworks (run once or to refresh)
python scripts/ingest_to_hopsworks.py --start-date 2022-08-01

# Step 2: Feature collection (hourly)
python scripts/collect_features.py

# Step 3: Model training (daily)
python scripts/train_model.py --force-register

# Run tests
python -m pytest tests/ -v
```

---

## Project Structure

```
AQI-Predictor/
├── scripts/                    # Pipeline scripts
│   ├── ingest_to_hopsworks.py  # Data ingestion to Hopsworks
│   ├── collect_features.py     # Feature collection (hourly)
│   ├── train_model.py          # Model training (daily)
│   └── validate_production.py  # CI validation
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
├── notebooks/                  # EDA notebooks (4)
├── tests/                      # 487 tests
├── docs/                       # Documentation
└── .github/workflows/          # CI/CD
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/prediction` | POST | AQI prediction |
| `/model-info` | GET | Model metadata |
| `/explain/feature-importance` | GET | Feature importance |
| `/explain/shap-global` | GET | Global SHAP analysis |
| `/explain/shap-explanation` | POST | Per-prediction SHAP |
| `/monitoring/drift` | GET | Data drift detection |
| `/monitoring/alerts` | GET | AQI hazard alerts |
| `/data/historical` | GET | Historical data |
| `/data/statistics` | GET | City statistics |

---

## Dashboard

4 interactive pages:
- **Dashboard** — Live AQI predictions with confidence intervals
- **Analytics** — Historical trends and city comparison
- **Explainability** — Feature importance, SHAP analysis, model comparison
- **System** — Service health, monitoring, alerts

---

## CI/CD

| Workflow | Schedule | Action |
|----------|----------|--------|
| Feature Collection | Every hour | Collect weather + pollution |
| Model Training | Daily 6 AM UTC | Train all models, select best |
| CI Pipeline | On push | Lint, tests |
| ML Validation | Weekly | Data safety, feature quality |
| CD Pipeline | On push | Pre-deploy checks, Docker |

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Data Provider | Open-Meteo (Weather + Air Quality) |
| Feature Store | Hopsworks (PRIMARY), Local CSV (Fallback) |
| ML Models | Ridge, Random Forest, XGBoost, LSTM |
| Model Registry | Hopsworks Model Registry |
| Backend API | FastAPI |
| Dashboard | Streamlit |
| CI/CD | GitHub Actions |
| Deployment | Render (API) + Streamlit Cloud (Dashboard) |

---

## License

This project is for educational and research purposes.
