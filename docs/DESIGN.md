# Design Document

## AQI Predictor — Design Decisions and Component Responsibilities

**Version:** 1.0  
**Date:** 31 July 2026  
**Status:** Phase 0 — Foundation  

---

## 1. Design Philosophy

The system is designed following these principles:

1. **Layered Separation** — Each layer has a single responsibility and communicates through well-defined interfaces
2. **Abstraction Before Implementation** — External dependencies are behind abstract interfaces to enable fallback and testing
3. **Fail Gracefully** — Every external call has retry, timeout, and fallback mechanisms
4. **Evidence-Based Decisions** — ML model selection is based on experimental comparison, not assumptions
5. **Extensibility** — The system supports multiple cities and can accommodate new data sources

---

## 2. Key Design Decisions

### 2.1 Multi-Output Forecast Design

**Decision:** Use a single multi-output model predicting `[AQI_24h, AQI_48h, AQI_72h]` simultaneously.

**Alternatives Considered:**
- **Three separate models** — One model per horizon (24h, 48h, 72h)
- **Recursive single model** — Predict 24h, then use that to predict 48h, etc.

**Reasoning:**
- A single multi-output model simplifies deployment and reduces operational complexity
- Recursive prediction compounds errors across horizons
- Three separate models triple the maintenance burden without clear accuracy benefit for this use case

**Impact:** Model must handle multi-output regression; evaluation metrics computed per-horizon and averaged.

---

### 2.2 Feature Store Abstraction

**Decision:** Abstract feature storage behind a `FeatureStoreInterface` with Hopsworks (primary) and DuckDB+Parquet (fallback) implementations.

**Alternatives Considered:**
- **Hopsworks only** — No fallback; simpler but fragile
- **DuckDB only** — No cloud feature store; loses MLOps maturity
- **Direct coupling** — Call Hopsworks directly in pipeline code

**Reasoning:**
- Hopsworks provides production-grade feature store capabilities but has reliability concerns (RPC disconnects, free-tier limits)
- Local fallback ensures development continuity when Hopsworks is unavailable
- Abstraction enables testing without cloud connectivity

**Impact:** All feature store operations go through the interface; implementations are interchangeable via configuration.

---

### 2.3 Data Source Priority

**Decision:** OpenWeather API as primary data source; AQICN/WAQI as fallback.

**Alternatives Considered:**
- **AQICN as primary** — Known to return static values for extended periods
- **Multiple APIs simultaneously** — Schema conflicts and synchronization issues

**Reasoning:**
- OpenWeather provides more frequent updates and consistent data
- AQICN has known staleness issues (e.g., returning same value for hours)
- Single primary source keeps schema consistent; fallback improves reliability

**Impact:** Pipeline logic detects OpenWeather failure and transparently switches to AQICN with staleness validation.

---

### 2.4 Model Experiment Framework

**Decision:** Train all four models (Ridge, Random Forest, XGBoost, LSTM) and select production model through evidence-based comparison.

**Selection Criteria:**
1. **Accuracy:** MAE, RMSE, R² (per-horizon)
2. **Inference Speed:** Latency per prediction
3. **Complexity:** Model size, training time, infrastructure requirements
4. **Maintainability:** Code complexity, debugging difficulty, deployment ease

**Impact:** All models tracked in MLflow; final selection documented in DECISIONS.md and MODEL_REPORT.md.

---

### 2.5 Time-Series Data Split

**Decision:** Chronological train/validation/test split with no random shuffling.

**Alternatives Considered:**
- **Random shuffle split** — Standard ML approach but leaks temporal information
- **Walk-forward validation** — More rigorous but computationally expensive

**Reasoning:**
- AQI forecasting is inherently temporal; future data must not leak into training
- Chronological split most closely mimics real deployment conditions
- Walk-forward validation can be added as an enhancement if needed

**Impact:** Split ratio and methodology documented in MODEL_REPORT.md for reproducibility.

---

### 2.6 API Backend Selection

**Decision:** FastAPI for the backend API.

**Alternatives Considered:**
- **Flask** — Mentioned in project-description.txt but FastAPI chosen in MASTER_AGENT_INSTRUCTIONS
- **Direct Streamlit model loading** — Bypasses API layer; loses separation of concerns

**Reasoning:**
- FastAPI provides automatic OpenAPI documentation
- Native async support for better performance
- Pydantic integration for request/response validation
- Clean separation between model serving and presentation

**Impact:** Streamlit dashboard communicates exclusively through FastAPI; no direct model loading in frontend.

---

### 2.7 City Extensibility

**Decision:** Architecture supports user-selected cities with initial set: Karachi, Lahore, Islamabad.

**Design:**
- City configuration stored in `config.yaml` with coordinates and metadata
- Feature store keys include `location_id` for city-specific feature groups
- API endpoints accept city as path parameter
- Dashboard provides city selection dropdown

**Impact:** Adding a new city requires only configuration entry and data collection setup; no code changes.

---

## 3. Component Responsibilities

### 3.1 src/data/

| File | Responsibility |
|---|---|
| `openweather_client.py` | API authentication, request construction, response parsing, retry logic for OpenWeather |
| `aqicn_client.py` | API authentication, request construction, response parsing, retry logic for AQICN |
| `validators.py` | Schema validation, staleness detection, duplicate prevention, data quality checks |
| `schemas.py` | Pydantic models defining API response schemas and internal data contracts |

### 3.2 src/features/

| File | Responsibility |
|---|---|
| `feature_engineering.py` | Transform raw data into ML-ready features (time, historical, derived) |
| `feature_validation.py` | Verify feature quality, detect data leakage, validate completeness |

### 3.3 src/models/

| File | Responsibility |
|---|---|
| `training.py` | Orchestrate model training with time-series splits, hyperparameters, reproducibility |
| `evaluation.py` | Compute metrics (MAE, RMSE, R²), generate comparison reports, create visualizations |
| `prediction.py` | Load production model, retrieve features, generate multi-output forecasts |

### 3.4 src/feature_store/

| File | Responsibility |
|---|---|
| `hopsworks_store.py` | Hopsworks connection, feature group management, insert/retrieve with retry |
| `local_store.py` | DuckDB/Parquet fallback for local feature storage and retrieval |

### 3.5 src/monitoring/

| File | Responsibility |
|---|---|
| `drift_detection.py` | Evidently AI integration, data drift reports, prediction monitoring |

### 3.6 app/

| File | Responsibility |
|---|---|
| `backend/fastapi_app.py` | API endpoints, request validation, model loading, prediction serving |
| `frontend/streamlit_app.py` | Interactive dashboard, charts, city selection, SHAP visualization |

### 3.7 pipelines/

| Directory | Responsibility |
|---|---|
| `feature_pipeline/` | End-to-end feature collection, engineering, and storage |
| `training_pipeline/` | End-to-end model training, evaluation, and registration |
| `monitoring_pipeline/` | Drift detection, report generation, alerting |

---

## 4. Error Handling Strategy

### 4.1 External API Failures
```
Detect failure → Retry (with backoff) → Log error → Use fallback API → Continue safely
```

### 4.2 Feature Store Failures
```
Detect failure → Retry (3 attempts) → Log error → Fall back to local store → Continue pipeline
```

### 4.3 Model Loading Failures
```
Detect failure → Log error → Return error response to dashboard → Display user-friendly message
```

### 4.4 General Principle
- Never silently ignore failures
- Always log with enough context to diagnose
- Prefer safe degradation over complete failure

---

## 5. Configuration Strategy

| File | Purpose |
|---|---|
| `.env` | API keys, connection strings (never committed) |
| `.env.example` | Placeholder templates for environment variables |
| `config.yaml` | Non-secret configuration (city coordinates, pipeline schedules, model parameters) |

**Key principle:** Code, configuration, and secrets are always separated.
