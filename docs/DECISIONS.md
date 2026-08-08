# Decisions

## AQI Predictor — Decision Documentation System

**Version:** 1.0  
**Date:** 31 July 2026  
**Status:** Phase 0 — Foundation  

---

## Decision Log

---

### DEC-001

**Date:** 31 July 2026  
**Topic:** Primary Python Version  
**Phase:** 0  

**Problem:**  
Hopsworks SDK has known incompatibilities with Python 3.12+ due to removal of the `imp` standard library module. Need to select a Python version that ensures compatibility across all dependencies.

**Options Considered:**

- **Option A:** Python 3.12 — Latest stable version, but breaks Hopsworks compatibility (`ModuleNotFoundError: No module named 'imp'`)
- **Option B:** Python 3.11 — Stable, well-supported, fully compatible with Hopsworks, TensorFlow, and all ML libraries
- **Option C:** Python 3.10 — Compatible but older; fewer modern language features

**Chosen Approach:** Python 3.11

**Reason:**  
Python 3.11 provides the best balance of modern language features and full compatibility with the locked technology stack (Hopsworks, TensorFlow, Scikit-learn, XGBoost). Python 3.12+ is explicitly forbidden by Hopsworks compatibility requirements.

**Trade-offs:**  
- Losing Python 3.12 features (minor)
- Gaining full Hopsworks and ecosystem compatibility (significant)

**Impact:**  
All development, testing, and deployment environments must use Python 3.11.

---

### DEC-002

**Date:** 31 July 2026  
**Topic:** Backend API Framework  
**Phase:** 0  

**Problem:**  
Need to select a backend framework for the prediction API. Project description mentions Flask, but master instructions specify FastAPI.

**Options Considered:**

- **Option A:** Flask — Lightweight, widely known, mentioned in project-description.txt
- **Option B:** FastAPI — Modern, automatic OpenAPI docs, native async, Pydantic integration, specified in MASTER_AGENT_INSTRUCTIONS.md

**Chosen Approach:** FastAPI

**Reason:**  
MASTER_AGENT_INSTRUCTIONS.md is the highest authority and explicitly specifies FastAPI. FastAPI provides superior features for ML serving: automatic API documentation, request/response validation via Pydantic, and native async support.

**Trade-offs:**  
- Slightly newer technology with less community content than Flask (minor)
- Significantly better API development experience and documentation (major)

**Impact:**  
Streamlit dashboard communicates with FastAPI backend; no direct model loading in frontend code.

---

### DEC-003

**Date:** 31 July 2026  
**Topic:** Primary Data Source  
**Phase:** 0  

**Problem:**  
Need to select primary and fallback external APIs for weather and air quality data collection.

**Options Considered:**

- **Option A:** AQICN as primary — Known to return static values for extended periods (e.g., AQI stuck at 161)
- **Option B:** OpenWeather as primary — More frequent updates, dynamic data, consistent API
- **Option C:** Multiple APIs simultaneously — Introduces schema conflicts and synchronization problems

**Chosen Approach:** OpenWeather as primary, AQICN as fallback

**Reason:**  
OpenWeather provides more frequent and dynamic data updates. AQICN has documented staleness issues. Single primary source keeps schema consistent while fallback improves reliability.

**Trade-offs:**  
- Two API integrations to maintain (necessary complexity)
- OpenWeather free-tier has rate limits (mitigated with caching and mock data during development)

**Impact:**  
Pipeline detects OpenWeather failure and transparently switches to AQICN with staleness validation.

---

### DEC-004

**Date:** 31 July 2026  
**Topic:** Feature Store Strategy  
**Phase:** 0  

**Problem:**  
Need feature storage that provides both cloud-grade capabilities and local development reliability.

**Options Considered:**

- **Option A:** Hopsworks only — Production-grade but fragile with RPC disconnects and free-tier limits
- **Option B:** DuckDB/Parquet only — No cloud feature store; loses MLOps maturity
- **Option C:** Hopsworks primary with DuckDB/Parquet fallback — Best of both worlds

**Chosen Approach:** Option C — Hopsworks primary with DuckDB/Parquet fallback

**Reason:**  
Hopsworks provides production-grade feature store capabilities required by the project. Local fallback ensures development continuity when Hopsworks is unavailable or rate-limited. Abstraction pattern enables clean switching.

**Trade-offs:**  
- Two implementations to maintain (necessary for reliability)
- Abstraction adds slight code complexity (justified by fallback benefit)

**Impact:**  
All feature store operations go through `FeatureStoreInterface`; implementations are interchangeable via configuration.

---

### DEC-005

**Date:** 31 July 2026  
**Topic:** Forecast Model Design  
**Phase:** 0  

**Problem:**  
Need to decide how to structure the 3-day AQI prediction (24h, 48h, 72h).

**Options Considered:**

- **Option A:** Single multi-output model — One model predicting all three horizons simultaneously
- **Option B:** Three separate models — One model per horizon
- **Option C:** Recursive prediction — Predict 24h, then use that to predict 48h, then 72h

**Chosen Approach:** Single multi-output model

**Reason:**  
Simplifies deployment and reduces operational complexity. Recursive prediction compounds errors. Three separate models triple maintenance burden without clear accuracy benefit.

**Trade-offs:**  
- Model must handle multi-output regression (implemented via sklearn MultiOutputRegressor or native XGBoost multi-output)
- Evaluation metrics computed per-horizon for detailed comparison

**Impact:**  
Model outputs `[AQI_24h, AQI_48h, AQI_72h]` as a single prediction vector.

---

### DEC-006

**Date:** 31 July 2026  
**Topic:** AQI Classification Standard  
**Phase:** 0  

**Problem:**  
Need a standard AQI category system for dashboard display and hazard alerts.

**Options Considered:**

- **Option A:** US EPA AQI (0-500 scale) — Widely documented, internationally recognized
- **Option B:** Pakistan NEQS standards — Locally relevant but less documented

**Chosen Approach:** US EPA AQI categories

**Reason:**  
US EPA AQI is the most widely documented and internationally recognized standard. Categories are well-defined with clear health recommendations.

**Trade-offs:**  
- Not locally specific to Pakistan (minor; can add local context in dashboard descriptions)

**Impact:**  
Dashboard alerts and classifications use US EPA AQI ranges (Good: 0-50, Moderate: 51-100, etc.).

---

### DEC-007

**Date:** 31 July 2026  
**Topic:** Initial Supported Cities  
**Phase:** 0  

**Problem:**  
Need to define initial cities for development and demonstration while keeping architecture extensible.

**Options Considered:**

- **Option A:** Single city — Faster to develop but less impressive demo
- **Option B:** Three cities — Good demo coverage; Karachi, Lahore, Islamabad are major Pakistani cities with varying air quality profiles

**Chosen Approach:** Karachi, Lahore, Islamabad as initial cities, with extensible architecture for future cities.

**Reason:**  
Three cities provide diverse air quality profiles for demonstration. Karachi (coastal, industrial), Lahore (inland, agricultural/industrial), Islamabad (capital, relatively cleaner). Architecture uses configuration-driven city support so new cities can be added via configuration without code changes.

**Trade-offs:**  
- Three cities triple the initial data collection scope (acceptable for demo quality)
- Extensible design requires abstracting city-specific logic (beneficial for long-term maintainability)

**Impact:**  
City coordinates stored in `config.yaml`; API calls and feature store use `location_id` for city-specific data. System is never hardcoded to specific cities — all city references are configuration-driven.

---

### DEC-008

**Date:** 31 July 2026  
**Topic:** Mock Data Volume and Usage  
**Phase:** 0  

**Problem:**  
How much synthetic data to generate for development and testing, and how to prevent misuse.

**Options Considered:**

- **Option A:** 30 days — May not be enough for feature engineering testing
- **Option B:** 90 days — Sufficient for pipeline validation and feature testing
- **Option C:** 180 days — More data but longer generation time

**Chosen Approach:** 90 days of mock data

**Reason:**  
90 days provides sufficient data volume for testing feature engineering (rolling averages, lag features), pipeline validation, and CI/CD testing. Enough to simulate seasonal patterns.

**Trade-offs:**  
- 90 days of synthetic data requires reasonable generation time (acceptable)
- Mock data must never be used for final model training or reported results

**Impact:**  
`data/mock/` contains 90 days of synthetic data. Documentation explicitly states mock data usage boundaries.

---

### DEC-009

**Date:** 1 August 2026  
**Topic:** TensorFlow Package Variant  
**Phase:** 1  

**Problem:**  
Full TensorFlow package includes GPU support and is significantly larger than needed for CPU-only development.

**Options Considered:**

- **Option A:** `tensorflow` — Full package with GPU support, large install size
- **Option B:** `tensorflow-cpu` — CPU-only variant, smaller install, faster setup

**Chosen Approach:** `tensorflow-cpu`

**Reason:**  
GPU support is not required for initial development and experimentation. CPU-only variant reduces install time and disk usage. Can be upgraded to full TensorFlow later if GPU training is needed.

**Trade-offs:**  
- No GPU acceleration during development (acceptable for model comparison phase)
- Significantly smaller installation footprint

**Impact:**  
`requirements.txt` specifies `tensorflow-cpu`. LSTM training will be slower on CPU but sufficient for experimentation.

---

### DEC-010

**Date:** 1 August 2026  
**Topic:** Hopsworks Host Configuration  
**Phase:** 1  

**Problem:**  
Hopsworks host URL should not be hardcoded in configuration files for security and flexibility.

**Options Considered:**

- **Option A:** Hardcode in config.yaml — Simple but inflexible
- **Option B:** Load from environment variable — Secure, flexible, follows 12-factor app principles

**Chosen Approach:** Environment variable `HOPSWORKS_HOST`

**Reason:**  
Environment variables are the standard for secret/sensitive configuration. Allows different hosts for development, staging, and production without code changes.

**Trade-offs:**  
- Requires .env file or system environment setup (standard practice)

**Impact:**  
`.env.example` includes `HOPSWORKS_HOST=eu-west.cloud.hopsworks.ai`. `config.yaml` does not contain host. All feature store code reads host from `os.environ`.

---

### DEC-011

**Date:** 1 August 2026  
**Topic:** Code Quality Pre-commit Hooks  
**Phase:** 1  

**Problem:**  
Need consistent code formatting and linting across the project.

**Options Considered:**

- **Option A:** No code quality tools — Inconsistent formatting
- **Option B:** Manual formatting — Error-prone, time-consuming
- **Option C:** Pre-commit hooks (black, isort, flake8) — Automated, consistent

**Chosen Approach:** Pre-commit hooks with black, isort, flake8

**Reason:**  
Industry-standard Python code quality tools. Black handles formatting, isort handles import sorting, flake8 handles linting. Pre-commit ensures checks run automatically.

**Trade-offs:**  
- Initial setup overhead (minimal)
- Developers must run `pre-commit install` once

**Impact:**  
`.pre-commit-config.yaml` created. All Python files should pass black, isort, and flake8 checks.

---

### DEC-012

**Date:** 2 August 2026  
**Topic:** API Client Credential Handling  
**Phase:** 2  

**Problem:**  
API clients should support both authenticated and unauthenticated modes for development and testing.

**Options Considered:**

- **Option A:** Require API key at initialization — Breaks test/mock mode
- **Option B:** Support optional API key with dependency injection — Flexible for all use cases

**Chosen Approach:** Optional API key with dependency injection

**Reason:**  
Allows clients to be instantiated without credentials for unit testing and mock data scenarios. API key is injected when available from environment variables.

**Trade-offs:**  
- Client must handle None API key gracefully (logged as warning)
- Tests can create clients without mocking credentials

**Impact:**  
All API clients accept `api_key=None`. Warning logged if initialized without key.

---

### DEC-013

**Date:** 2 August 2026  
**Topic:** API Retry Strategy  
**Phase:** 2  

**Problem:**  
Need to define which errors trigger retries and which fail immediately.

**Options Considered:**

- **Option A:** Retry all errors — Wastes time on permanent failures
- **Option B:** Retry only transient errors — Efficient and correct

**Chosen Approach:** Retry only transient errors

**Reason:**  
Authentication failures (401/403) and invalid requests (4xx) are permanent and will not succeed on retry. Network failures, timeouts, rate limits (429), and server errors (5xx) are transient.

**Trade-offs:**  
- More complex error classification (justified by correctness)
- No wasted retry attempts on permanent failures

**Impact:**  
`BaseAPIClient._retry_request()` classifies errors and retries only: `APINetworkError`, `APITimeoutError`, `APIRateLimitError`, `APIServerError`.

---

### DEC-014

**Date:** 2 August 2026  
**Topic:** Data Source Ownership  
**Phase:** 2  

**Problem:**  
Both OpenWeather and AQICN provide overlapping data. Need clear ownership rules.

**Options Considered:**

- **Option A:** Use whichever source is available — Unpredictable data quality
- **Option B:** Define authoritative source per field category — Consistent, predictable

**Chosen Approach:** Define authoritative source per field category

**Reason:**  
Weather fields are more accurate from OpenWeather (specialized weather API). AQI/pollution values from AQICN use the US EPA scale and are more reliable for air quality assessment.

**Trade-offs:**  
- Requires merge logic between sources (implemented in AQICNClient.merge_with_openweather)
- Clear, documented data provenance

**Impact:**  
OpenWeather is authoritative for temperature, humidity, wind, pressure, weather_condition. AQICN is authoritative for AQI, PM2.5, PM10, CO, NO2, SO2, O3.

---

### DEC-015

**Date:** 8 August 2026  
**Topic:** Synthetic Data Usage Restrictions  
**Phase:** 5  

**Problem:**  
Historical API data is not yet available. Pipeline development requires data for testing, but synthetic data must not contaminate final results.

**Options Considered:**

- **Option A:** Use synthetic data for everything — Fast but invalidates model results
- **Option B:** Use synthetic data only for pipeline testing; real data for training — Correct but slower
- **Option C:** Wait for real data before any development — Too slow

**Chosen Approach:** Option B — Synthetic data restricted to pipeline testing only

**Reason:**  
Synthetic data allows pipeline architecture to be validated without waiting for API credentials. However, model training and evaluation must use only real API data to produce valid, reproducible results.

**Constraints:**  
Synthetic data CANNOT be used for:
- Model training
- Model evaluation
- Reported metrics
- Production use

Synthetic data CAN be used for:
- Pipeline validation
- Unit testing
- Integration testing
- Architecture verification

**Trade-offs:**  
- Final model results must wait for real data (correct)
- Pipeline can be developed and tested in parallel (efficient)

**Impact:**  
All dataset metadata includes `approved_for_training: false` for synthetic datasets. Real data collection must complete before Phase 7 model training produces reported results.
