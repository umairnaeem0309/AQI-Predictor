# 🌍 AQI Predictor

**Predict Air Quality Index 24/48/72 hours ahead for Pakistani cities using Machine Learning.**

[![CI](https://github.com/umairnaeem0309/AQI-Predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/umairnaeem0309/AQI-Predictor/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-487%20passed-green)]()
[![Python](https://img.shields.io/badge/python-3.11-blue)]()

[Live Dashboard](https://airpulse.streamlit.app/) | [API Backend](https://aqi-predictor-api-nf7s.onrender.com) | [Documentation](docs/)

---

## Overview

A production-grade ML pipeline that collects real-time weather and air quality data from Open-Meteo, engineers 63 features, trains multiple models (Ridge, Random Forest, XGBoost, LSTM), and serves predictions via a REST API and interactive dashboard.

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

## Verified Model Performance

*All results verified on 4-year dataset (Aug 2022 – Aug 2026).*

### Production Model: XGBoost

| Metric | Value |
|--------|-------|
| **Test MAE** | **21.34** |
| **Test RMSE** | **30.35** |
| **Test R²** | **0.6584** |
| **Train Time** | 9.9s |
| **Inference Latency** | 0.011 ms/sample |

### Model Comparison — Test Set

| Model | MAE | RMSE | R² | Latency |
|-------|-----|------|----|---------|
| **XGBoost** | **21.34** | **30.35** | **0.6584** | 0.011 ms |
| Random Forest | 21.61 | 30.58 | 0.6533 | 0.013 ms |
| Ridge | 21.73 | 30.64 | 0.6520 | 0.0003 ms |
| LSTM | 22.95 | 32.46 | 0.6092 | 0.057 ms |

---

## Verified Data Pipeline

| Stage | Status | Details |
|-------|--------|---------|
| Data Collection | ✅ Verified | Open-Meteo API, 4-year range |
| Data Cleaning | ✅ Verified | 107,208 rows, 0 duplicates, <0.2% NaN |
| Feature Engineering | ✅ Verified | 63 features |
| Feature Store | ✅ Verified | Hopsworks: 107,208 rows |
| Model Training | ✅ Verified | All 4 models on complete data |
| Model Registry | ✅ Verified | Hopsworks Model Registry |
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
# Feature collection (hourly)
python scripts/collect_features.py

# Model training (daily)
python scripts/train_model.py --force-register

# Run tests
python -m pytest tests/ -v
```

---

## Project Structure

```
AQI-Predictor/
├── scripts/                    # Pipeline scripts
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
| Feature Store | Hopsworks (PRIMARY), Local Parquet (Fallback) |
| ML Models | Ridge, Random Forest, XGBoost, LSTM |
| Model Registry | Hopsworks Model Registry |
| Backend API | FastAPI |
| Dashboard | Streamlit |
| CI/CD | GitHub Actions |
| Deployment | Render (API) + Streamlit Cloud (Dashboard) |

---

## License

This project is for educational and research purposes.
