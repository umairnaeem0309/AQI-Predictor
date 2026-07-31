# Architecture Document

## AQI Predictor — System Architecture

**Version:** 1.0  
**Date:** 31 July 2026  
**Status:** Phase 0 — Foundation  

---

## 1. Overview

The AQI Predictor is a production-grade MLOps system that forecasts Air Quality Index at 24h, 48h, and 72h horizons. The architecture follows a layered design with clear separation of concerns between data collection, feature engineering, model management, serving, and presentation.

---

## 2. High-Level Architecture

```
                        ┌─────────────────┐
                        │      User       │
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │   Streamlit     │
                        │   Dashboard     │
                        │  (Frontend)     │
                        └────────┬────────┘
                                 │ HTTP
                        ┌────────▼────────┐
                        │    FastAPI      │
                        │    Backend      │
                        └────────┬────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                   │
     ┌────────▼────────┐ ┌──────▼──────┐  ┌────────▼────────┐
     │   Prediction    │ │  MLflow     │  │  Evidently AI   │
     │   Service       │ │  Model      │  │  Monitoring     │
     │                 │ │  Registry   │  │                 │
     └────────┬────────┘ └──────┬──────┘  └─────────────────┘
              │                  │
              │      ┌───────────┘
              │      │
     ┌────────▼──────▼────────┐
     │   Feature Store        │
     │  ┌───────────────────┐ │
     │  │ Hopsworks (Primary)│ │
     │  │ DuckDB+Parquet    │ │
     │  │ (Fallback)        │ │
     │  └───────────────────┘ │
     └───────────┬────────────┘
                 │
     ┌───────────▼────────────┐
     │   Data Pipeline Layer  │
     │  ┌───────────────────┐ │
     │  │ Data Validation   │ │
     │  │ Feature Engineer  │ │
     │  │ Data Collection   │ │
     │  └───────────────────┘ │
     └───────────┬────────────┘
                 │
     ┌───────────▼────────────┐
     │   External APIs        │
     │  ┌───────────────────┐ │
     │  │ OpenWeather (Prim)│ │
     │  │ AQICN (Fallback)  │ │
     │  └───────────────────┘ │
     └────────────────────────┘
```

---

## 3. Component Descriptions

### 3.1 Data Collection Layer

| Component | Purpose | Technology |
|---|---|---|
| OpenWeather Client | Primary weather + air quality data ingestion | Python HTTP client |
| AQICN Client | Fallback data source when OpenWeather fails | Python HTTP client |
| Validators | Schema validation, staleness detection, duplicate prevention | Pandas + Pydantic |
| Schemas | API response schemas and data contracts | Pydantic models |

**Responsibilities:**
- Make authenticated API requests with retry logic
- Parse and normalize API responses into standard DataFrames
- Validate data quality before passing downstream
- Detect and handle stale or duplicate data
- Log all API interactions for auditability

### 3.2 Feature Engineering Layer

| Component | Purpose |
|---|---|
| Feature Engineering | Transform raw data into ML-ready features |
| Feature Validation | Verify feature quality and detect leakage |

**Feature Categories:**
- **Time-based:** hour, day, month, weekday, season (encoded)
- **Historical:** lag values (1h, 6h, 12h, 24h, 48h, 72h), rolling averages (6h, 12h, 24h, 72h)
- **Derived:** AQI change rate, pollutant ratios (PM2.5/PM10, NO2/SO2), weather interaction terms

**Critical Rule:** Every feature must have a documented data availability time to prevent leakage.

### 3.3 Feature Storage Layer

| Component | Technology | Role |
|---|---|---|
| Hopsworks Store | Hopsworks Feature Store (Hudi format) | Primary cloud feature storage |
| Local Store | DuckDB + Parquet | Fallback for development and offline use |

**Design Pattern:** Abstract `FeatureStoreInterface` with two implementations.

```
FeatureStoreInterface (abstract)
    │
    ├── HopsworksStore (primary)
    │     Uses: eu-west.cloud.hopsworks.ai
    │     Format: Apache Hudi
    │     Retry: Yes, with exponential backoff
    │
    └── LocalStore (fallback)
          Uses: DuckDB queries on Parquet files
          Location: data/processed/
```

### 3.4 ML Training Layer

| Component | Purpose |
|---|---|
| Training Module | Orchestrate model training with time-series splits |
| Evaluation Module | Compute MAE, RMSE, R² and generate comparison reports |
| Prediction Module | Load production model and generate multi-output forecasts |

**Models to Compare:**

| Model | Purpose | Framework |
|---|---|---|
| Ridge Regression | Linear baseline | Scikit-learn |
| Random Forest | Tree-based baseline | Scikit-learn |
| XGBoost | Advanced tabular model | XGBoost |
| LSTM | Sequential deep learning | TensorFlow/Keras |

**Target Design:** Single multi-output model predicting `[AQI_24h, AQI_48h, AQI_72h]`.

### 3.5 Experiment Tracking & Model Registry

| Component | Technology |
|---|---|
| Experiment Tracking | MLflow Tracking |
| Model Registry | MLflow Model Registry |
| Artifact Storage | MLflow artifact store (local/filesystem) |

**Every registered model contains:**
- Model name and version
- Training date and timestamp
- Dataset version and feature version
- Model parameters and hyperparameters
- Evaluation metrics (MAE, RMSE, R²)
- Training environment information
- Random seed for reproducibility

### 3.6 API Backend

| Component | Technology |
|---|---|
| Backend Framework | FastAPI |
| Request Validation | Pydantic models |
| API Documentation | Auto-generated OpenAPI/Swagger |

**Endpoints:**

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Service availability check |
| GET | `/health` | Health check with model and feature store status |
| GET | `/prediction/{city}` | 3-day AQI forecast for specified city |
| GET | `/features/{city}` | Current feature values for a city |
| GET | `/model-info` | Production model metadata and version |

**Response Schema (Prediction):**
```json
{
  "city": "Karachi",
  "timestamp": "2026-08-15T10:00:00Z",
  "aqi_24h": 142,
  "aqi_48h": 138,
  "aqi_72h": 145,
  "category": "Unhealthy for Sensitive Groups",
  "model_version": "3"
}
```

### 3.7 Dashboard

| Component | Technology |
|---|---|
| Dashboard Framework | Streamlit |
| Data Source | FastAPI backend |

**Dashboard Sections:**
1. **Main Dashboard** — Current AQI, category, 3-day forecast, forecast chart
2. **Analytics Dashboard** — Historical AQI trends, pollutant trends, weather relationships
3. **Explainability Dashboard** — SHAP feature importance, model explanation
4. **System Dashboard** — Model version, last training time, data freshness, pipeline status

### 3.8 Monitoring

| Component | Technology |
|---|---|
| Drift Detection | Evidently AI |
| Data Drift | Feature distribution monitoring |
| Prediction Monitoring | Trend analysis, anomaly detection |

### 3.9 Automation

| Workflow | Technology | Trigger |
|---|---|---|
| Testing | GitHub Actions | Push, Pull Request |
| Feature Pipeline | GitHub Actions | Every 6h (dev), every 1h (prod) |
| Training Pipeline | GitHub Actions | Daily |

### 3.10 Application Database

| Component | Technology | Purpose |
|---|---|---|
| Application DB | SQLite | Prediction history, application metadata |

SQLite is **not** a replacement for the feature store. It handles lightweight application-level storage only.

---

## 4. Data Flow

### 4.1 Data Collection Flow
```
OpenWeather API / AQICN API
        │
        ▼
API Client (with retry + fallback)
        │
        ▼
Schema Validation (Pydantic)
        │
        ▼
Staleness Check + Deduplication
        │
        ▼
Raw Data Storage (data/raw/)
```

### 4.2 Feature Engineering Flow
```
Raw Data
    │
    ▼
Time Feature Extraction (hour, day, month, weekday, season)
    │
    ▼
Historical Feature Computation (lags, rolling averages)
    │
    ▼
Derived Feature Calculation (ratios, change rates, interactions)
    │
    ▼
Feature Validation (leakage check, missing values)
    │
    ▼
Feature Store Insertion (Hopsworks or Local)
```

### 4.3 Prediction Flow
```
User selects city in Streamlit
    │
    ▼
Dashboard calls FastAPI /prediction/{city}
    │
    ▼
FastAPI loads production model from MLflow
    │
    ▼
FastAPI retrieves latest features from Feature Store
    │
    ▼
Model generates [AQI_24h, AQI_48h, AQI_72h]
    │
    ▼
Response with predictions + category + model version
    │
    ▼
Dashboard displays forecast + charts + SHAP explanation
```

---

## 5. Infrastructure

### 5.1 Development Environment
- **OS:** Windows
- **IDE:** VS Code
- **Python:** 3.11 (venv)
- **Package Manager:** pip

### 5.2 Containerization
- **Docker** for reproducible builds
- **docker-compose.yml** for multi-service orchestration

### 5.3 Deployment Targets
- **Frontend:** Streamlit Cloud
- **Backend:** Cloud hosting compatible with FastAPI

---

## 6. Directory Structure

```
AQI-Predictor/
│
├── app/
│   ├── frontend/
│   │   └── streamlit_app.py
│   └── backend/
│       └── fastapi_app.py
│
├── src/
│   ├── data/
│   │   ├── openweather_client.py
│   │   ├── aqicn_client.py
│   │   ├── validators.py
│   │   └── schemas.py
│   ├── features/
│   │   ├── feature_engineering.py
│   │   └── feature_validation.py
│   ├── models/
│   │   ├── training.py
│   │   ├── evaluation.py
│   │   └── prediction.py
│   ├── feature_store/
│   │   ├── hopsworks_store.py
│   │   └── local_store.py
│   ├── monitoring/
│   │   └── drift_detection.py
│   ├── utils/
│   └── config/
│
├── pipelines/
│   ├── feature_pipeline/
│   ├── training_pipeline/
│   └── monitoring_pipeline/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── mock/
│
├── models/
├── notebooks/
├── research/
│   ├── EDA/
│   ├── feature_analysis/
│   └── model_comparison/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── end_to_end/
│
├── docs/
├── docker/
├── .github/workflows/
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── README.md
└── .gitignore
```
