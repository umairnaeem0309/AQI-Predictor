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

### DEC-014 (Amended)

**Date:** 2 August 2026 | **Amended:** 26 August 2026  
**Topic:** Data Source Authority  
**Phase:** 2 | **Amended in:** Phase 17  

**Problem:**  
Both OpenWeather and AQICN provide overlapping data. Need clear ownership rules for weather, AQI, and pollutant fields.

**Amendment trigger:**  
AQICN stations for Karachi, Lahore, and Islamabad are confirmed stale (data months/years old). AQICN cannot provide training-quality AQI observations for Pakistani cities.

**Weather:**  
OpenWeather remains authoritative for temperature, humidity, wind, pressure, and weather conditions.

**External US EPA AQI:**  
AQICN remains the preferred external AQI source when the selected station is geographically valid and the observation satisfies the approved freshness requirement.

**Pakistan operational condition:**  
The validated AQICN stations for Karachi, Lahore, and Islamabad are currently too stale to provide training-quality AQI observations.

**Fallback target:**  
When AQICN is unavailable or stale, the project derives a US EPA-method particle-pollution NowCast AQI from OpenWeather PM2.5 and PM10 hourly concentrations.

For each timestamp:
- Calculate PM2.5 NowCast AQI when valid
- Calculate PM10 NowCast AQI when valid
- Select the higher valid sub-index
- Record its pollutant as dominant

**Derived data description:**  
This derived value is described as a "US EPA-method PM NowCast AQI derived from OpenWeather pollutant concentrations". It is NOT an official EPA/AirNow monitor observation and is not yet a complete multi-pollutant AQI because O3, CO, SO2, and NO2 sub-indices are not included in the Phase 17 target.

**OpenWeather main.aqi:**  
The OpenWeather 1-5 index is never substituted for this project's 0-500 AQI target.

**Methodology:**  
EPA-454/B-24-002, May 2024.

**Trade-offs:**  
- Requires merge logic between sources (implemented in AQICNClient.merge_with_openweather)
- Derived AQI introduces estimation uncertainty vs official monitor readings
- Clear, documented data provenance and full metadata trail

**Impact:**  
OpenWeather is authoritative for temperature, humidity, wind, pressure, weather_condition, PM2.5, PM10. AQI target is derived from OpenWeather pollutants using EPA PM NowCast methodology when AQICN is stale.

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

---

### DEC-016

**Date:** 20 August 2026  
**Topic:** Production Deployment Strategy  
**Phase:** 14  

**Problem:**  
Need a production deployment strategy that ensures safety, reliability, and rollback capability.

**Options Considered:**

- **Option A:** Manual deployment — Simple but error-prone
- **Option B:** Docker Compose production deployment — Containerized, reproducible
- **Option C:** Kubernetes deployment — Enterprise-grade but overkill for current scale

**Chosen Approach:** Docker Compose production deployment

**Reason:**  
Docker Compose provides containerized, reproducible deployments without Kubernetes complexity. Sufficient for current scale. Multi-stage builds ensure minimal image size. Python-based healthchecks eliminate curl dependency.

**Constraints:**  
- MOCK_MODE must be false in production
- Only approved real-data models may be deployed
- Pre-deployment safety checks are mandatory
- Rollback strategy must be automated

**Trade-offs:**  
- Docker Compose has limited scaling compared to Kubernetes (acceptable for current needs)
- Requires Docker and Docker Compose on deployment host (standard)

**Impact:**  
Production deployment uses `docker/docker-compose.prod.yml`. All deployments run `pre_deploy_checks.py` before deployment. gunicorn with uvicorn workers serves the FastAPI backend.

---

### DEC-017

**Date:** 26 August 2026  
**Topic:** AQICN Station Selection Strategy  
**Phase:** 17  

**Problem:**  
AQICN city-level feeds (`/v2/feed/{city}/`) return severely stale data (timestamps from months ago), while bound station IDs (`/v2/@{station_id}/`) return fresh, current observations.

**Options Considered:**

- **Option A:** Continue using city-level feeds — Simple but data is stale and training-invalid
- **Option B:** Switch to bound station IDs — Requires mapping cities to station IDs but returns fresh data
- **Option C:** Switch AQI provider entirely — Requires project-owner approval, not within Phase 17 scope

**Chosen Approach:** Option B — Use AQICN bound station IDs

**Reason:**  
Bound station IDs provide genuinely fresh AQI observations. City-level feeds are fundamentally stale. Station mapping is a one-time configuration task.

**Constraints:**  
- Bound station IDs: Karachi=@7393, Lahore=@7432, Islamabad=@7433
- Source freshness must be validated using provider observation timestamp, not collection time
- Observations with stale AQI values are marked training-invalid

**Trade-offs:**  
- Requires maintaining station ID mapping (minimal effort)
- Station IDs may change if AQICN reconfigures (monitor periodically)
- AQICN ground stations update infrequently (~6-7 hour intervals)

**Impact:**  
AQICN client uses bound station IDs for fresh data. Quality gate validates source observation freshness. Training dataset excludes stale observations.

---

### DEC-018

**Date:** 27 August 2026  
**Topic:** Historical Data Source Migration to Open-Meteo  
**Phase:** 17 (Revision)  

**Problem:**  
The previous Phase 17 strategy depended on OpenWeather + AQICN + 30-day live collection. Timeline constraints require an immediate historical dataset without waiting 30 days. AQICN Pakistan stations are confirmed stale. OpenWeather historical weather requires a paid subscription.

**Options Considered:**

- **Option A:** Continue 30-day live collection — Correct but too slow for timeline
- **Option B:** Open-Meteo Historical Weather + Air Quality APIs — Free, no API key, 5+ years hourly data
- **Option C:** Purchase OpenWeather historical access — Paid, single source
- **Option D:** Switch to another paid AQI provider — Requires new credentials, untested

**Chosen Approach:** Option B — Open-Meteo Historical APIs

**Reason:**  
Open-Meteo provides free, no-API-key-required access to:
- Historical Weather: hourly data from 2017+ (IFS 9km) or 1940+ (ERA5)
- Air Quality: hourly CAMS data from Aug 2022+ (Global) or 2013+ (European)

This enables immediate generation of a 4-5 year dataset without waiting, paid subscriptions, or API key management.

**Data sources:**  
- Weather: `/v1/archive` endpoint — temperature, humidity, pressure, wind, cloud cover, precipitation
- Air Quality: `/v1/air-quality` endpoint — PM2.5, PM10, CO, NO2, SO2, O3
- US AQI: also available from Open-Meteo for validation; project uses own EPA calculation

**AQI methodology preserved:**  
US EPA PM AQI calculated from Open-Meteo PM2.5 and PM10 concentrations using existing EPA-454/B-24-002 May 2024 breakpoints. The project does NOT use Open-Meteo's built-in US AQI values as the prediction target.

**Provider abstraction maintained:**  
New Open-Meteo providers implement `BaseHistoricalProvider`. Existing OpenWeather/AQICN real-time collection remains functional for future live inference.

**Constraints:**  
- Open-Meteo weather: IFS 9km from 2017+, ERA5 from 1940+
- Open-Meteo air quality: CAMS Global from Aug 2022+ (45km resolution)
- Effective overlap for both weather + AQ: Aug 2022+
- Free tier: non-commercial use only, no API key required
- Rate limits: generous (sub-second response, chunked requests for large ranges)

**Trade-offs:**  
- CAMS Global AQ is 45km resolution (coarser than ground monitors) — acceptable for city-level modeling
- Air quality data starts Aug 2022 (not full 5 years) — 4 years of overlap is sufficient
- Reanalysis weather data may differ slightly from ground station observations — documented
- No real-time data from Open-Meteo for production inference (existing OpenWeather/AQICN retained)

**Impact:**  
- New provider classes: `OpenMeteoWeatherProvider`, `OpenMeteoAirQualityProvider`
- New pipeline: `historical_ingestion.py` for batch download, merge, AQI calculation
- New CLI: `scripts/build_dataset.py` for dataset generation
- Existing real-time collection (`api_manager.py`, `real_data_collector.py`) unchanged
- Dataset output: `data/processed/{train,val,test}_{features,targets}.csv`
- Configuration added to `config.yaml` under `api.open_meteo`

**DEC-014 relationship:**  
DEC-014 (Data Source Authority) is amended to include Open-Meteo as the historical data source. For real-time inference, OpenWeather and AQICN remain the primary and fallback sources per DEC-014.
