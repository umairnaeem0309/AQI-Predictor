# 🌍 AQI Predictor

**Predict Air Quality Index 24/48/72 hours ahead for Pakistani cities using Machine Learning.**

[![CI](https://github.com/umairnaeem0309/AQI-Predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/umairnaeem0309/AQI-Predictor/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-487%20passed-green)]()
[![Python](https://img.shields.io/badge/python-3.11-blue)]()

[Live Dashboard](https://airpulse.streamlit.app/) | [API Backend](https://aqi-predictor-api-nf7s.onrender.com) | [Documentation](docs/)

---

## Overview

A production-grade ML pipeline that collects real-time weather and air quality data from Open-Meteo, engineers 68 features, trains multiple models (Ridge, Random Forest, XGBoost, LSTM), and serves predictions via a REST API and interactive dashboard.

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
│ 68 features: weather, pollution, time, lags, rolling, ratios │
│ → src/features/feature_engineering.py                        │
└─────────────────────────────┬───────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ FEATURE STORE (Hopsworks PRIMARY)                            │
│ → Hopsworks cloud (63,648 rows)                              │
│ → Local Parquet (fallback)                                   │
└─────────────────────────────┬───────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ MODEL TRAINING (Daily)                                       │
│ Ridge, Random Forest, XGBoost, LSTM                          │
│ → scripts/train_model.py                                     │
└─────────────────────────────┬───────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ MODEL REGISTRY (MLflow)                                      │
│ Version tracking, metrics, rollback                          │
│ → src/models/registry.py                                     │
└─────────────────────────────┬───────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ DEPLOYMENT                                                   │
│ FastAPI Backend (Render) + Streamlit Dashboard (Cloud)       │
└─────────────────────────────────────────────────────────────┘
```

---

## Verified Model Performance

*All results verified on full dataset (2026-08-31).*

### Model Comparison — Overall (Test Set)

| Model | MAE | RMSE | R² | Inference Latency |
|-------|-----|------|----|-------------------|
| **Random Forest** | **22.59** | **30.37** | **0.6281** | 0.048 ms/sample |
| Ridge Regression | 23.49 | 31.20 | 0.6077 | 0.001 ms/sample |
| XGBoost | 23.45 | 31.38 | 0.6031 | 0.012 ms/sample |
| LSTM | 23.97 | 31.95 | 0.5882 | 0.159 ms/sample |

### Per-Horizon Breakdown (Test Set)

| Horizon | Best Model | MAE | RMSE | R² |
|---------|------------|-----|------|----|
| 24h | Random Forest | 19.39 | 26.93 | 0.7016 |
| 48h | Random Forest | 23.36 | 30.83 | 0.6135 |
| 72h | Random Forest | 25.00 | 33.03 | 0.5692 |

**Selected Model:** Random Forest (composite score: 0.4×MAE + 0.3×RMSE + 0.3×(1-R²)×100)

---

## Verified Data Pipeline

| Stage | Status | Details |
|-------|--------|---------|
| Data Collection | ✅ Verified | Live Open-Meteo API for all 3 cities |
| Data Cleaning | ✅ Verified | 107K rows, 0 duplicates, <0.2% NaN |
| Feature Engineering | ✅ Verified | 68 features created correctly |
| Feature Store | ✅ Verified | Hopsworks: 63,648 rows stored & read |
| Model Training | ✅ Verified | All 4 models trained on complete data |
| Model Registry | ✅ Verified | MLflow tracking experiments |
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

# Backfill Hopsworks
python scripts/backfill_hopsworks.py

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
│   ├── backfill_hopsworks.py   # Hopsworks backfill
│   └── validate_production.py  # CI validation
│
├── src/                        # Source code
│   ├── data/                   # Data collection
│   │   ├── live_fetcher.py     # Live data fetcher
│   │   └── providers/          # API providers
│   ├── features/               # Feature engineering
│   ├── feature_store/          # Hopsworks + Local
│   ├── models/                 # ML models
│   └── monitoring/             # Drift detection
│
├── app/                        # Web application
│   ├── backend/                # FastAPI (15 endpoints)
│   └── frontend/               # Streamlit (4 pages)
│
├── notebooks/                  # EDA notebooks
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
- **Explainability** — Feature importance, SHAP analysis
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
| Model Registry | MLflow (local) |
| Backend API | FastAPI |
| Dashboard | Streamlit |
| CI/CD | GitHub Actions |
| Deployment | Render (API) + Streamlit Cloud (Dashboard) |

---

## License

This project is for educational and research purposes.
