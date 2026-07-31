# Phases Document

## AQI Predictor — Complete Project Phases

**Version:** 1.0  
**Date:** 31 July 2026  
**Status:** Phase 0 — Foundation  

---

## Phase Overview

The project consists of **17 phases** (Phase 0 through Phase 16). Each phase must be completed, tested, documented, committed, and approved before the next phase begins.

| Phase | Name | Status |
|---|---|---|
| 0 | Requirement Analysis and Project Foundation | 🔄 In Progress |
| 1 | Repository and Development Environment Setup | ⏳ Pending |
| 2 | Data Collection Architecture | ⏳ Pending |
| 3 | Real API Integration | ⏳ Pending |
| 4 | Feature Engineering Pipeline | ⏳ Pending |
| 5 | Historical Data Backfill | ⏳ Pending |
| 6 | Feature Store Implementation | ⏳ Pending |
| 7 | Machine Learning Experiment Pipeline | ⏳ Pending |
| 8 | Model Selection and Production Model Decision | ⏳ Pending |
| 9 | MLflow Model Registry Implementation | ⏳ Pending |
| 10 | Automation with GitHub Actions | ⏳ Pending |
| 11 | Monitoring Implementation | ⏳ Pending |
| 12 | FastAPI Backend Implementation | ⏳ Pending |
| 13 | Streamlit Dashboard Implementation | ⏳ Pending |
| 14 | Deployment | ⏳ Pending |
| 15 | Final Documentation and Project Delivery | ⏳ Pending |
| 16 | Demo Preparation and Final Review | ⏳ Pending |

---

## Phase 0 — Requirement Analysis and Project Foundation

**Objective:** Understand the complete project before writing implementation code.

**Scope:** Documentation only — no production code.

**Tasks:**
- Analyze all requirements from source documents
- Create project documentation foundation (15 docs files)
- Confirm architecture understanding
- Identify risks and ambiguities
- Define initial project roadmap

**Files Expected:**
```
docs/PRD.md
docs/ARCHITECTURE.md
docs/DESIGN.md
docs/RULES.md
docs/PHASES.md
docs/PLAN.md
docs/CURRENT_STATE.md
docs/MEMORY.md
docs/DECISIONS.md
docs/PROJECT_JOURNAL.md
```

**Testing:** Documentation review only.

**Completion Criteria:**
- [ ] Project plan exists
- [ ] Architecture is documented
- [ ] Requirements are understood
- [ ] No unresolved major ambiguity

---

## Phase 1 — Repository and Development Environment Setup

**Objective:** Create the professional engineering foundation.

**Tasks:**
- Create repository directory structure
- Set up Python 3.11 virtual environment
- Create `requirements.txt` with pinned dependencies
- Create `Dockerfile` and `docker-compose.yml`
- Create `.env.example` and `.gitignore`
- Set up logging framework
- Set up testing framework (pytest)

**Files Expected:**
```
requirements.txt
Dockerfile
docker-compose.yml
.env.example
.gitignore
src/ (directory skeleton)
tests/ (directory skeleton)
```

**Testing:**
- Python version check
- Dependency import test
- Configuration loading test

**Completion Criteria:**
- [ ] Repository structure created
- [ ] Virtual environment works with Python 3.11
- [ ] Dependencies installed successfully
- [ ] Docker configuration functional
- [ ] Tests framework operational

---

## Phase 2 — Data Collection Architecture

**Objective:** Build the data ingestion foundation using mock data.

**Tasks:**
- Implement `src/data/openweather_client.py`
- Implement `src/data/aqicn_client.py`
- Implement `src/data/validators.py`
- Implement `src/data/schemas.py`
- Create mock data in `data/mock/`
- Write unit tests for API response parsing, validation, error handling

**Testing:** API response parsing, schema validation, retry behavior.

**Completion Criteria:**
- [ ] API abstraction layer complete
- [ ] Mock data collection works end-to-end
- [ ] Validation and error handling tested

---

## Phase 3 — Real API Integration

**Objective:** Connect to real external data sources.

**Required Credentials:** `OPENWEATHER_API_KEY`, `AQICN_API_KEY`

**Tasks:**
- Integrate OpenWeather API
- Integrate AQICN fallback API
- Implement staleness detection and deduplication
- Store API responses in `data/raw/api_audit/`
- Test with real API responses

**Testing:** Successful API connection, invalid API handling, missing fields, stale responses, duplicate prevention.

**Completion Criteria:**
- [ ] Real API data collected successfully
- [ ] Fallback mechanism works
- [ ] Data quality checks pass

---

## Phase 4 — Feature Engineering Pipeline

**Objective:** Transform raw data into ML-ready features.

**Tasks:**
- Implement time-based features (hour, day, month, weekday, season)
- Implement historical features (lag values, rolling averages)
- Implement derived features (AQI change rate, pollutant ratios, weather interactions)
- Document feature leakage prevention
- Create `docs/DATA_DICTIONARY.md`

**Testing:** Correct calculations, missing value handling, edge cases.

**Completion Criteria:**
- [ ] All feature categories implemented
- [ ] Feature validation passes
- [ ] Data dictionary complete

---

## Phase 5 — Historical Data Backfill

**Objective:** Generate training dataset.

**Tasks:**
- Run feature pipeline over historical dates
- Generate features + targets dataset
- Document date range, record count, locations, data quality
- Validate no duplicates, missing values, timestamp consistency

**Testing:** Data quality validation.

**Completion Criteria:**
- [ ] Training dataset generated
- [ ] Data quality passes all checks
- [ ] Dataset metadata documented

---

## Phase 6 — Feature Store Implementation

**Objective:** Implement feature storage layer.

**Required Credential:** `HOPSWORKS_API_KEY`

**Tasks:**
- Implement `FeatureStoreInterface` abstraction
- Implement `src/feature_store/hopsworks_store.py` (Hudi format, eu-west endpoint)
- Implement `src/feature_store/local_store.py` (DuckDB + Parquet)
- Implement retry logic and fallback behavior
- Configure for Python 3.11 compatibility

**Testing:** Feature creation, insertion, retrieval, fallback behavior.

**Completion Criteria:**
- [ ] Hopsworks integration works
- [ ] Local fallback works
- [ ] Retry logic tested

---

## Phase 7 — Machine Learning Experiment Pipeline

**Objective:** Train and compare forecasting models.

**Tasks:**
- Implement Ridge Regression training
- Implement Random Forest training
- Implement XGBoost training
- Implement LSTM training (TensorFlow/Keras)
- Time-series train/validation/test split
- Track all experiments in MLflow
- Compute MAE, RMSE, R² per model and per horizon

**Testing:** Training pipeline runs, models save correctly, metrics generate.

**Completion Criteria:**
- [ ] All four models trained successfully
- [ ] Metrics computed and compared
- [ ] All experiments tracked in MLflow

---

## Phase 8 — Model Selection and Production Model Decision

**Objective:** Select final production model using experimental evidence.

**Tasks:**
- Compare all model results (MAE, RMSE, R²)
- Compare engineering metrics (speed, complexity, maintainability)
- Document decision in `docs/DECISIONS.md` and `docs/MODEL_REPORT.md`
- Justify selection with evidence

**Completion Criteria:**
- [ ] Production model selected with documented reasoning
- [ ] Rejected models explained
- [ ] MODEL_REPORT.md complete

---

## Phase 9 — MLflow Model Registry Implementation

**Objective:** Create professional model lifecycle management.

**Tasks:**
- Implement experiment tracking
- Implement artifact logging
- Implement model registration and versioning
- Implement model loading for prediction

**Testing:** Model registration, retrieval, prediction loading.

**Completion Criteria:**
- [ ] Model lifecycle works end-to-end
- [ ] Metadata recorded for every model

---

## Phase 10 — Automation with GitHub Actions

**Objective:** Automate project workflows.

**Tasks:**
- Create testing workflow (push/PR triggered)
- Create feature pipeline workflow (every 6h dev, every 1h prod)
- Create training workflow (daily)
- Handle failures, produce logs, protect secrets

**Testing:** Workflow execution, environment setup, test completion.

**Completion Criteria:**
- [ ] All three workflows operational
- [ ] Failures handled gracefully

---

## Phase 11 — Monitoring Implementation

**Objective:** Monitor ML system health.

**Tasks:**
- Implement Evidently AI data drift monitoring
- Implement prediction monitoring
- Generate monitoring reports and alerts

**Completion Criteria:**
- [ ] Drift detection operational
- [ ] Monitoring reports generated

---

## Phase 12 — FastAPI Backend Implementation

**Objective:** Create prediction API.

**Tasks:**
- Implement GET `/` and `/health` endpoints
- Implement GET `/prediction/{city}` endpoint
- Implement GET `/features/{city}` endpoint
- Implement GET `/model-info` endpoint
- Request validation, error handling, logging, API documentation

**Testing:** Endpoint tests, invalid request tests, model loading tests.

**Completion Criteria:**
- [ ] All endpoints functional
- [ ] API documentation generated
- [ ] Error handling tested

---

## Phase 13 — Streamlit Dashboard Implementation

**Objective:** Create interactive user interface.

**Tasks:**
- Implement Main Dashboard (current AQI, 3-day forecast, chart)
- Implement Analytics Dashboard (historical trends, pollutant trends)
- Implement Explainability Dashboard (SHAP feature importance)
- Implement System Dashboard (model version, pipeline status)

**Testing:** UI loads, API connection works, charts render.

**Completion Criteria:**
- [ ] All dashboard sections functional
- [ ] Dashboard consumes FastAPI correctly

---

## Phase 14 — Deployment

**Objective:** Deploy the complete system.

**Tasks:**
- Deploy frontend to Streamlit Cloud
- Deploy backend to cloud hosting
- Verify health checks and model loading

**Completion Criteria:**
- [ ] Deployed dashboard accessible
- [ ] API responds correctly
- [ ] Model loads and predicts

---

## Phase 15 — Final Documentation and Project Delivery

**Objective:** Prepare final professional project package.

**Tasks:**
- Complete README.md with architecture diagram, screenshots, instructions
- Verify all 15 documentation files are complete
- Create final report covering architecture, data pipeline, ML pipeline, deployment, challenges

**Completion Criteria:**
- [ ] All documentation complete
- [ ] README.md comprehensive
- [ ] Final checklist verified

---

## Phase 16 — Demo Preparation and Final Review

**Objective:** Prepare for presentation and external evaluation.

**Tasks:**
- Verify fresh installation works
- Capture final screenshots (dashboard, MLflow, GitHub Actions, monitoring)
- Create demo checklist and walkthrough

**Completion Criteria:**
- [ ] Demo flow is reproducible
- [ ] All screenshots captured
- [ ] Final approval received
