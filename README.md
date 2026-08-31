# 🌍 AQI Predictor

**Predict Air Quality Index 24/48/72 hours ahead for Pakistani cities using Machine Learning.**

[![CI](https://github.com/umairnaeem0309/AQI-Predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/umairnaeem0309/AQI-Predictor/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-487%20passed-green)]()
[![Python](https://img.shields.io/badge/python-3.11-blue)]()

[Live Dashboard](https://airpulse.streamlit.app/) | [API Backend](https://aqi-predictor-api-nf7s.onrender.com) | [Documentation](docs/)

---

## Overview

A production-grade ML pipeline that collects real-time weather and air quality data from Open-Meteo, engineers 63+ features, trains multiple models (Ridge, Random Forest, XGBoost, LSTM), and serves predictions via a REST API and interactive dashboard.

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
│ 63+ features: weather, pollution, time, lags, rolling, ratios│
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

*All results verified on complete 4-year dataset (63,504 usable rows).*

### Production Model: Ridge Regression

| Metric | Value |
|--------|-------|
| **Test MAE** | **26.48** |
| **Test RMSE** | **34.95** |
| **Test R²** | **0.5722** |
| **Composite Score** | **33.91** |
| **Inference Latency** | 0.000 ms |

### Model Comparison — Test Set (All Models)

| Model | MAE | RMSE | R² | Composite | Latency |
|-------|-----|------|----|-----------|---------|
| **Ridge** | **26.48** | **34.95** | **0.5722** | **33.91** ★ | 0.000 ms |
| Random Forest | 27.24 | 35.80 | 0.5510 | 35.11 | 0.009 ms |
| XGBoost | 28.18 | 37.26 | 0.5136 | 37.04 | 0.015 ms |

### Per-Horizon Comparison — Test Set

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

### Why Ridge?

1. **Lowest MAE** (26.48) across all models
2. **Highest R²** (0.5722) — explains most variance
3. **Wins ALL 3 horizons** — consistent performance
4. **Fastest inference** (0.000 ms) — production-ready
5. **Most interpretable** — linear coefficients directly explain feature influence
6. **Least overfitting risk** — simple model with regularization

---

## Verified Data Pipeline

| Stage | Status | Details |
|-------|--------|---------|
| Data Collection | ✅ Verified | Open-Meteo API, 4-year range |
| Data Cleaning | ✅ Verified | 107,208 rows, 0 duplicates, <0.2% NaN |
| Feature Engineering | ✅ Verified | 63 features |
| Feature Store | ✅ Verified | Hopsworks: 107,208 rows |
| Model Training | ✅ Verified | All models on complete data |
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
