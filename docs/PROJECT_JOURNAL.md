# Project Journal

## AQI Predictor — Chronological Engineering History

**Version:** 1.0  
**First Entry:** 31 July 2026  

---

## Entry 001

**Date:** 31 July 2026  
**Phase:** Phase 0 — Requirement Analysis and Project Foundation  

### Work Completed
- Read and analyzed MASTER_AGENT_INSTRUCTIONS.md (4,169 lines) — the highest authority document
- Read and analyzed all 6 source documents:
  - `project-description.txt` — Project overview and technology stack
  - `issues_with_the_workflow.md` — Verified technical issues and solutions
  - `hops_work_setup_prompt.md` — Hopsworks MLOps setup best practices
  - `answers to students questions from hafsa.md` — Mentor Q&A guidance
  - `AQI_predict-1.pdf` — Project PDF (binary document; accessible information extracted via metadata; full text not available)
  - `The-Complete-Machine-Learning-Pipeline.pptx` — ML pipeline presentation (binary document; accessible information extracted via metadata; full text not available)
- Confirmed full project understanding
- Created Phase 0 planning output with 7 sections
- Received project owner approval with decisions on:
  - City support: Karachi, Lahore, Islamabad (extensible)
  - AQI standard: US EPA categories
  - Mock data: 90 days for testing only
  - Git timeline: 31 July 2026 start date
  - Hopsworks: Account setup guidance in Phase 6
- Created all 10 documentation foundation files:
  - `docs/PRD.md` — Product Requirements Document
  - `docs/ARCHITECTURE.md` — System Architecture
  - `docs/DESIGN.md` — Design Decisions and Component Responsibilities
  - `docs/RULES.md` — Development Rules and Constraints
  - `docs/PHASES.md` — Complete Project Phases (17 phases)
  - `docs/PLAN.md` — Project Roadmap and Timeline
  - `docs/CURRENT_STATE.md` — Current Project Status
  - `docs/MEMORY.md` — Long-Term Project Knowledge
  - `docs/DECISIONS.md` — Decision Documentation System (8 decisions)
  - `docs/PROJECT_JOURNAL.md` — This journal

### Problems
- Two source documents are binary (PDF and PPTX); accessible information was extracted from available metadata and context, and binary document limitations were documented
- No significant technical issues encountered during documentation phase

### Solutions
- Binary document limitations documented; all requirements are fully captured through other source documents and MASTER_AGENT_INSTRUCTIONS.md
- All requirements are fully captured in the documentation foundation

### Decisions Made
- 8 pre-approved decisions documented in DECISIONS.md (DEC-001 through DEC-008)
- All decisions align with MASTER_AGENT_INSTRUCTIONS.md and project owner guidance

### Next Step
- Complete Phase 0 with completion report
- Wait for Phase 0 approval
- Begin Phase 1 — Repository and Development Environment Setup

---

## Entry 002

**Date:** 1 August 2026  
**Phase:** Phase 1 — Repository and Development Environment Setup  

### Work Completed
- Created complete repository directory structure with __init__.py and .gitkeep files
- Created requirements.txt with pinned dependencies (tensorflow-cpu preferred)
- Created .env.example with HOPSWORKS_HOST environment variable (not in config.yaml)
- Created config.yaml with city definitions, model parameters, API settings (no hardcoded Hopsworks host)
- Created .gitignore with comprehensive rules for secrets, Python, data, models, IDE files
- Created Dockerfile (Python 3.11-slim) and docker-compose.yml (backend + frontend services)
- Created .pre-commit-config.yaml (black, isort, flake8)
- Created src/config/__init__.py with setup_logging() and load_config()
- Created tests/conftest.py and tests/unit/test_environment.py with 5 test classes
- Created placeholder source files for all modules with docstrings
- Updated CURRENT_STATE.md and PROJECT_JOURNAL.md

### Files Added
```
requirements.txt
.env.example
config.yaml
.gitignore
Dockerfile
docker-compose.yml
.pre-commit-config.yaml
src/config/__init__.py
tests/conftest.py
tests/unit/test_environment.py
+ placeholder source files (12 files)
+ directory skeleton (30+ directories with __init__.py/.gitkeep)
```

### Problems
- Hopsworks host moved to env var per owner request (no code impact)
- tensorflow-cpu preferred over full tensorflow to reduce install size

### Solutions
- HOPSWORKS_HOST in .env.example; config.yaml references env var
- requirements.txt specifies tensorflow-cpu

### Decisions Made
- DEC-009: Use tensorflow-cpu by default
- DEC-010: Hopsworks host from env var, not config file
- DEC-011: Pre-commit hooks: black, isort, flake8

### Next Step
- Run environment verification tests
- Create Phase 1 git commit
- Wait for Phase 1 approval

---

## Entry 003

**Date:** 2 August 2026  
**Phase:** Phase 2 — Data Collection Architecture  

### Work Completed
- Created src/data/exceptions.py with custom exception hierarchy (12 exception classes)
- Created src/data/base_client.py with abstract base, retry logic, caching readiness
- Created src/data/schemas.py with 15 Pydantic models for API responses and standard schema
- Created src/data/validators.py with full validation pipeline (schema, staleness, duplicates, missing)
- Created src/data/openweather_client.py with weather + pollution merging and timezone normalization
- Created src/data/aqicn_client.py with staleness detection and AQICN-OpenWeather merge
- Created 7 mock API response JSON files (API-shaped responses only)
- Created 5 test files: test_schemas, test_openweather_client, test_aqicn_client, test_validators, test_retry_logic
- Created docs/DATA_DICTIONARY.md with comprehensive field documentation
- Added responses library to requirements.txt
- Updated CURRENT_STATE.md and MEMORY.md

### Key Design Decisions
- Data ownership: OpenWeather authoritative for weather, AQICN authoritative for AQI/pollution
- API clients support initialization without credentials (dependency injection)
- Retry only on: network failures, timeouts, HTTP 429, HTTP 5xx
- No retry on: authentication failures, invalid requests
- Timezone normalization: all timestamps converted to UTC
- Mock data: API-shaped responses only, never fake training datasets

### Problems
- None significant during implementation

### Decisions Made
- DEC-012: API clients support initialization without credentials
- DEC-013: Retry only on retryable errors (network, timeout, 429, 5xx)
- DEC-014: OpenWeather authoritative for weather; AQICN authoritative for AQI/pollution

### Next Step
- Create Phase 2 git commit
- Wait for Phase 2 approval

---

## Entry 004

**Date:** 8 August 2026  
**Phase:** Phase 5 — Synthetic Data Clarification  

### Work Completed
- Renamed synthetic data references to "mock historical dataset" throughout codebase
- Added dataset metadata: `dataset_type: synthetic_test_data`, `approved_for_training: false`, `approved_for_evaluation: false`
- Updated CURRENT_STATE.md with critical synthetic data restriction notice
- Added DEC-015: Synthetic Data Usage Restrictions to DECISIONS.md

### Critical Clarification
- The 4,392 training observations are **synthetic test data**, not real API data
- No real historical API calls were made (no credentials provided)
- Synthetic data is approved for pipeline testing ONLY
- Final model training and evaluation require real API data

### Decisions Made
- DEC-015: Synthetic data restricted to pipeline testing only

### Next Step
- Proceed to Phase 6 (Feature Store) with synthetic data for pipeline validation only
- Real data collection must complete before Phase 7 reported results

---

## Entry 005

**Date:** 15 August 2026  
**Phase:** Phase 9 — Model Lifecycle Management  

### Work Completed
- Created `src/models/lifecycle.py` with lifecycle state machine:
  - 8 model statuses: UNTRAINED, TRAINING, EVALUATED, CANDIDATE, APPROVED, REGISTERED, PRODUCTION, ARCHIVED, REJECTED
  - Valid transition graph with enforcement
  - `validate_lifecycle_transition()` for transition checking
- Extended `src/models/registry.py` with:
  - `store_version_metadata()` for comprehensive version tracking
  - `get_drift_baseline()` for drift detection reference
  - `load_production_model()` for model loading with validation
  - `_validate_metadata_for_load()` for metadata completeness checks
- Created `tests/unit/test_lifecycle.py` with 24 tests
- Created `tests/integration/test_model_loading.py` with 12 tests
- Updated CURRENT_STATE.md and PROJECT_JOURNAL.md

### Key Design Decisions
- Lifecycle transitions are strictly enforced; invalid transitions raise errors
- Production loading validates: status, approval, dataset_type, feature_version, schema_version
- Drift baseline includes: mean, std, min, max, percentiles for numerical features
- Model loading supports: current production, specific version, rollback target
- Synthetic data blocks lifecycle advancement to REGISTERED/PRODUCTION at approval level

### Problems
- None significant during implementation

### Decisions Made
- Model lifecycle transitions must be strictly validated
- Drift baseline stored as JSON with complete statistical summary
- Production model loading requires full metadata validation

### Next Step
- Review Phase 9 commits
- Approve Phase 10 for CI/CD pipeline implementation

---

## Entry 006

**Date:** 16 August 2026  
**Phase:** Phase 10 — CI/CD Pipeline  

### Work Completed
- Created `.github/workflows/ci.yml` with:
  - Python 3.11 matrix
  - pip dependency caching
  - Lint, type check, unit tests stages
  - Docker build verification
  - CI validation tests
  - Security audit
  - GitHub Actions permissions configuration
  - Artifact retention policy (30 days)
- Created `.github/workflows/ml-validation.yml` with:
  - Data safety validation
  - Model artifact validation
  - Feature quality checks
  - Lifecycle transition validation
- Created `.github/workflows/cd.yml` with:
  - Pre-deployment checks
  - Docker image build
  - Dry-run staging deployment
  - Dry-run production deployment
  - Deployment record creation
- Created `scripts/validate_production.py` reusing Phase 8/9 logic:
  - Production status validation
  - Approval status checks
  - Real API data requirement
  - Feature/schema version validation
  - Dataset approval flag verification
- Created `tests/ci/test_ci_validation.py` with tests for:
  - Synthetic data rejection
  - Missing secrets handling
  - Invalid model state rejection
  - Lifecycle validation
  - Registry safety checks
- Updated CURRENT_STATE.md and PROJECT_JOURNAL.md

### Key Design Decisions
- CI pipeline runs on all pushes/PRs; integration tests only on main
- ML validation runs weekly and on manual trigger
- CD workflow uses dry-run deployment until infrastructure exists
- Production validation reuses existing Phase 8/9 validation logic
- Credential-dependent tests skip gracefully when secrets unavailable
- pip-audit uses requirements.txt (no hash pinning)
- Artifact retention set to 30 days

### Problems
- None significant during implementation

### Decisions Made
- CI/CD pipeline uses GitHub Actions
- Dry-run deployment for staging and production
- Production validation script centralizes safety checks
- CI validation tests verify synthetic data rejection

### Next Step
- Review Phase 10 commits
- Approve Phase 11 for monitoring implementation

---

## Entry 007

**Date:** 17 August 2026  
**Phase:** Phase 11 — Monitoring Implementation  

### Work Completed
- Created `src/monitoring/__init__.py` with module exports
- Created `src/monitoring/drift_detection.py` with Evidently 0.7.21 integration:
  - DriftDetector class with PSI and KS test support
  - DriftResult and DriftReport dataclasses
  - Report generation, save, and load functionality
- Created `src/monitoring/performance.py`:
  - PerformanceMonitor with rolling metrics (24h, 7d, 30d)
  - Degradation detection with configurable thresholds
  - PerformanceMetric and PerformanceReport dataclasses
- Created `src/monitoring/alerting.py`:
  - AlertManager with cooldown and aggregation
  - Alert levels: INFO, WARNING, CRITICAL, EMERGENCY
  - Alert types: DATA_DRIFT, MODEL_PERFORMANCE, DATA_QUALITY, SYSTEM
  - Alert acknowledgement and resolution
- Created `src/monitoring/notification.py`:
  - LogNotifier for Python logging
  - ConsoleNotifier with color support
  - WebhookNotifier placeholder for future
- Created `src/monitoring/prediction_logger.py`:
  - JSONL storage format
  - Security checks for sensitive data
  - Feature hashing for privacy
  - Feedback loop support
- Created `src/monitoring/baseline_manager.py`:
  - Multiple baseline types (training, rolling, city-specific)
  - Synthetic data rejection for monitoring baselines
  - Baseline versioning and statistics
- Updated `requirements.txt` with Evidently 0.7.x
- Created unit tests: test_drift_detection.py, test_performance_monitor.py, test_alerting.py
- Created integration test: test_monitoring.py
- Updated CURRENT_STATE.md and PROJECT_JOURNAL.md

### Key Design Decisions
- Evidently version: 0.7.21 (compatible with Python 3.11)
- Separated testing_mock_data from synthetic_test_data
- Notification abstraction: Log and Console only (email/Slack deferred)
- Alert cooldown: 60 minutes default, configurable
- Prediction logging: JSONL format (database migration deferred)
- Security: No secrets, API keys, or PII in prediction logs
- Baseline rejection: synthetic_test_data blocked for monitoring
- Monitoring metadata: dataset_type, baseline_version, feature_version, model_version

### Problems
- None significant during implementation

### Decisions Made
- Evidently 0.7.x for drift detection
- JSONL for prediction logging (portable, simple)
- Security checks prevent sensitive data in logs
- Synthetic data rejected for monitoring baselines
- Alert cooldown prevents notification flooding

### Next Step
- Review Phase 11 commits
- Approve Phase 12 for FastAPI backend implementation

---

## Entry 008

**Date:** 20 August 2026  
**Phase:** Phase 14 — Deployment  

### Work Completed
- Created `docker/Dockerfile.backend` with:
  - Multi-stage build for optimized image size
  - Non-root user for security
  - Python-based healthcheck (no curl dependency)
  - gunicorn + uvicorn workers (4 workers)
- Created `docker/Dockerfile.frontend` with:
  - Multi-stage build
  - Streamlit configuration
  - Python-based healthcheck
- Created `docker/docker-compose.prod.yml` with:
  - Backend and frontend services
  - Health checks with start period
  - Resource limits
  - Network isolation
  - Volume persistence
- Created `scripts/pre_deploy_checks.py`:
  - MOCK_MODE validation (must be false)
  - API_KEY validation
  - HOPSWORKS_HOST validation
  - API keys validation
  - Model metadata validation
- Created `scripts/deploy.py`:
  - Full deployment automation
  - Pre-deployment checks
  - Backup creation
  - Rollback on failure
  - Deployment logging
- Created `tests/deployment/test_deployment_safety.py`:
  - Mock mode rejection tests
  - Synthetic model rejection tests
  - Missing secret rejection tests
  - Health failure rollback simulation tests
- Created `docs/DEPLOYMENT.md`:
  - Complete deployment guide
  - Environment variables documentation
  - Deployment steps
  - Rollback procedures
  - Troubleshooting guide
- Updated CURRENT_STATE.md and PROJECT_JOURNAL.md

### Key Design Decisions
- Production deployment target: Docker Compose
- Server strategy: gunicorn + uvicorn workers
- Healthcheck: Python-based (no curl dependency)
- MOCK_MODE=false enforced in pre-deployment checks
- Hopsworks local fallback warned, not silently activated
- Rollback automated on health check failure

### Problems
- None significant during implementation

### Decisions Made
- DEC-016: Production Deployment Strategy documented
- Python-based healthcheck eliminates curl dependency
- gunicorn + uvicorn chosen for production server

### Next Step
- Review Phase 14 commits
- Approve Phase 15 for final documentation

---

## Entry 009

**Date:** 21 August 2026  
**Phase:** Phase 15 — Final Documentation  

### Work Completed
- Added DEC-016: Production Deployment Strategy to DECISIONS.md
- Created `docs/PRODUCTION_READINESS.md`:
  - Pre-deployment checklist
  - Deployment checklist
  - Health check verification
  - Rollback procedure
  - Monitoring checklist
  - Troubleshooting guide
  - Sign-off section
- Updated CURRENT_STATE.md with Phase 15 status
- Updated PROJECT_JOURNAL.md with Phase 15 entry

### Key Design Decisions
- Production readiness checklist covers all phases
- Sign-off section for formal approval
- Troubleshooting guide for common issues

### Problems
- None significant during implementation

### Decisions Made
- Final documentation includes production readiness checklist
- Sign-off process defined for deployment approval

### Next Step
- Review Phase 15 commits
- Approve Phase 16 for demo preparation

---

## Entry 010

**Date:** 26 August 2026  
**Phase:** Phase 17 — Real Data Validation (Continuation)  

### Work Completed
- Created Python 3.11.15 conda environment (`aqi-predictor`)
- Installed all pinned dependencies within approved ranges
- Validated Hopsworks cloud connection (Feature Store accessible)
- Validated DuckDB/Parquet local fallback
- Investigated AQICN staleness: city-level feeds stale, bound stations fresh
- Fixed AQICN client to use bound station IDs for fresh data
- Added source-level freshness validation (collected_at, weather_observed_at, aqi_observed_at)
- Fixed quality gate to use provider observation timestamps
- Marked initial stale observations as training-invalid
- Updated requirements.txt to match verified working versions
- Fixed 8 test failures (import bugs, mock URL mismatches, response library API)
- Ran full test suite: 287 passed, 26 failed (pre-existing), 1 skipped

### Key Findings
- AQICN bound stations (@7393, @7432, @7433) return genuinely fresh data
- AQICN city-level feeds return stale data (months old timestamps)
- Hourly collection cadence feasible within API rate limits
- Historical data not available on tested free-tier API endpoints
- Python 3.11 environment resolves all Hopsworks/dependency issues

### Commits Created
1. `fix: improve AQICN freshness validation and station selection` (13 files)
2. `fix: correct quality gate to use source observation timestamps` (2 files)
3. `test: fix unit tests for AQICN and OpenWeather clients` (3 files)
4. `chore: update requirements.txt and add Python 3.11 environment scripts` (6 files)
5. `docs: document Phase 17 corrections and findings` (1 file)

### Files Modified
- src/data/aqicn_client.py — bound station IDs, freshness validation
- src/data/schemas.py — collected_at field, training_valid flag, data_source enum
- src/data/openweather_client.py — collected_at field
- src/data/base_client.py — missing import fix
- src/models/registry.py — missing imports fix
- scripts/quality_gate.py — source timestamp freshness, training-valid filtering
- scripts/validate_api.py — Windows encoding fix
- tests/unit/test_aqicn_client.py — bound station URL mocks
- tests/unit/test_openweather_client.py — timeout test fix
- tests/unit/test_api_validation.py — assertion fixes
- tests/unit/test_retry_logic.py — base_url, timeout fixes
- requirements.txt — updated pinned versions

### Problems
- Python 3.12.9 was active; project requires 3.11 for Hopsworks
- Hopsworks client v3.7.0 incompatible with backend v5.0.3
- AQICN city-level feeds return stale data; bound stations required
- Quality gate freshness was using collection time, not provider timestamp
- Multiple test failures due to evolving implementation vs stale mocks

### Solutions
- Created conda Python 3.11.15 environment
- Upgraded hopsworks to >=4.0.0 to match backend
- Switched AQICN client to bound station IDs
- Added source-level timestamp tracking to schema and clients
- Fixed quality gate to validate provider observation freshness
- Updated all affected tests to match new implementation

### Decisions Made
- Use AQICN bound station IDs, not city-level feeds, for fresh data
- Source freshness validation uses provider observation timestamp
- Initial stale observations marked training-invalid but preserved for audit
- Hourly collection cadence approved (within API limits)

### Next Step
- Begin sustained hourly real data collection
- Monitor AQICN freshness over multiple collection cycles
- After 21+ days, run quality gate for training readiness
- Wait for project-owner approval before Phase 18

---

## Entry 011

**Date:** 2 September 2026  
**Phase:** Final — Pipeline Verification and Documentation  

### Work Completed
- Verified Render API health: model loaded, feature store connected
- Verified predictions for all 3 cities (Karachi: 62, Lahore: 156, Islamabad: 111)
- Verified Hopsworks Feature Store: 107,064 rows, 64 columns, 3 cities
- Verified Hopsworks Model Registry: XGBoost v4 registered
- Fixed YAML syntax errors in feature-collection.yml and daily-training.yml
- Fixed Hopsworks Model Registry model loading (download() + pickle)
- Computed baseline comparisons (Mean Predictor, Persistence Model)
- Updated all documentation with baseline results
- Fixed CRLF line endings and added .gitattributes
- All 487 tests passing

### Key Findings
- XGBoost is 47% better than Mean Predictor (MAE 21.31 vs 40.42)
- XGBoost is 19% better than Persistence Model (MAE 21.31 vs 26.38)
- GitHub Actions YAML had syntax errors in embedded Python reporting (fixed)
- Hopsworks Model Registry uses download() + pickle, not load()

### Commits Created
1. `fix: fix YAML workflow syntax, Hopsworks model loading, and update all docs`

### Files Modified
- .github/workflows/feature-collection.yml — Fixed YAML syntax
- .github/workflows/daily-training.yml — Fixed YAML syntax
- scripts/report_collection.py — NEW: Collection health reporter
- scripts/report_training.py — NEW: Training results reporter
- src/models/hopsworks_registry.py — Fixed download() + pickle loading
- README.md — Updated with full project details and baselines
- docs/CURRENT_STATE.md — Updated with latest results and baselines
- docs/PROJECT_JOURNEY.md — Updated with complete journey and baselines
- docs/FINAL_PROJECT_REPORT.md — Updated with baselines
- docs/DECISIONS.md — Added DEC-021 baseline comparison
- docs/ARCHITECTURE.md — Fixed feature count, date

### Next Step
- Verify CI/CD pipeline passes after push
- Monitor daily training automation
- Project complete and ready for submission

---

## Entry 012

**Date:** 5 September 2026  
**Phase:** Operations — Live Collection Verification and Storage Cleanup  

### Work Completed
- Diagnosed hourly collector: GitHub Actions job ran every hour but ALL Hopsworks inserts were silently rejected (`Features are not compatible with Feature Group schema`) — proven by Hopsworks containing exactly 107,064 rows ending 2026-08-28 with zero live rows
- Queried the live `aqi_features_prod` v1 feature-group schema from Hopsworks (64 columns) and aligned `scripts/collect_features.py` to it exactly, including per-column types (double→float64, bigint→int64 for humidity/wind_direction/cloud_cover/season, int→int32 for hour/day_of_week/month/is_weekend)
- Changed collector timestamps to floored UTC hours so Hopsworks upserts on (location_id, timestamp) deduplicate retries within the same hour
- Added target backfill to `scripts/train_model.py`: 24/48/72h targets recomputed from the AQI series so live rows become trainable once future hours accumulate (deterministic, no leakage)
- Fixed daily-training.yml scheduled-run bug: empty `--min-improvement` argument on schedule events now defaults to 0.01
- Removed the local Parquet backup (`hourly_observations.parquet`, `collection_metadata.json`) from the collector AND from disk — Hopsworks is now the SINGLE data store (DEC-022)
- Updated workflows so `HOPSWORKS_PROJECT` can be a repository variable (non-sensitive) while host/API key remain repository secrets
- Verified with REAL API rounds: Hopsworks grew 107,064 → 107,067; read-back confirmed correct raw values, lags, rolling and time features; duplicate protection confirmed (re-run in same hour kept 107,067)
- Ran full test suite: 487 passed, 1 skipped
- Updated ALL documentation (README, CURRENT_STATE, PROJECT_JOURNEY, FINAL_PROJECT_REPORT, DECISIONS, ARCHITECTURE, DATASET_REPORT) and fixed inconsistencies: missing LSTM rows in 24h/48h per-horizon tables, stale DEC-020 metrics (21.34/0.6584/9.9s → 21.31/0.6588/23.7s), stale DATASET_REPORT counts (107,208/63 features → 107,064/58 features), README train-time mismatch (22.5s → 23.7s)

### Key Findings
- Hopsworks rejects inserts on type mismatches even when column names match — the schema must be matched exactly (including int32 vs int64 vs float64)
- Open-Meteo history already covers the current hour, so the collector must stamp current-observation values onto the engineered row explicitly
- GitHub scheduled workflows receive EMPTY `github.event.inputs.*` — defaults must be handled with `|| 'default'` expressions
- Daily training failure root cause: `HOPSWORKS_HOST`/`HOPSWORKS_API_KEY` repository secrets not configured in GitHub (env resolves to empty in Actions)

### Decisions Made
- DEC-022: Hopsworks is the single store for live collection; no local parquet backup
- Per-column type casting derived from the live Hopsworks feature-group schema (authoritative source)

### Files Modified
- scripts/collect_features.py — schema alignment, typed casts, UTC hour buckets, local backup removed
- scripts/train_model.py — target backfill from AQI series
- .github/workflows/daily-training.yml — min-improvement default, HOPSWORKS_PROJECT vars fallback
- .github/workflows/feature-collection.yml — HOPSWORKS_PROJECT vars fallback
- README.md + 7 docs — consistency fixes and live-collection verification

### Action Required (Owner)
- Add GitHub repository secrets `HOPSWORKS_API_KEY`, `HOPSWORKS_HOST` and repository variable `HOPSWORKS_PROJECT` (Settings → Secrets and variables → Actions) — cannot be done locally without GitHub authentication

### Next Step
- After secrets are added, verify the next hourly GitHub Actions run persists to Hopsworks and the next daily training completes
