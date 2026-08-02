# Memory

## AQI Predictor — Long-Term Project Knowledge

**Version:** 1.0  
**Date:** 2 August 2026  
**Status:** Phase 2 — Data Collection Architecture  

---

## 1. Technical Discoveries

### Hopsworks Platform Constraints
- Python 3.12+ is incompatible due to removal of `imp` module in Python 3.12
- Free-tier accounts have compute budget limits ($10+ usage can trigger account freezing)
- Frequent writes (hourly) exhaust free-tier credits rapidly
- RPC listener disconnects are a known issue with Delta Lake format
- Apache Hudi format provides more stable feature insertion
- Regional endpoint `eu-west.cloud.hopsworks.ai` is more reliable than default `c.app.hopsworks.ai`

### API Data Quality
- AQICN API returns static values for extended periods (e.g., same AQI reading for 6+ hours)
- OpenWeather API provides more frequent updates and dynamic data
- Staleness detection is critical when using AQICN as fallback
- Deduplication by `(timestamp, location_id)` prevents duplicate feature records

### Python Environment
- Python 3.11 is the safe choice for Hopsworks compatibility
- Python 3.10 is acceptable as fallback
- Python 3.12+ must not be used in this project

### Data Collection Architecture
- OpenWeather API provides both weather and air pollution data
- AQICN/WAQI API provides AQI on US EPA scale (0-500)
- OpenWeather AQI is 1-5 scale (different from US EPA) — not used as primary AQI
- AQICN ground stations update infrequently (staleness detection critical)
- All timestamps normalized to UTC for consistency
- API clients support initialization without credentials for testing
- Retry only on transient errors: network, timeout, 429, 5xx
- No retry on: 401, 403, 4xx (permanent failures)

---

## 2. Lessons Learned

### Phase 2 — Data Collection
- Pydantic v2 models provide strong type validation for API responses
- responses library is effective for HTTP mocking in tests
- Mock data should be API-shaped JSON responses only (never fake training data)
- Data ownership rules prevent conflicting values from multiple sources

---

## 3. Environment Information

### Development Environment
- Windows + VS Code
- Python 3.11 virtual environment
- pip for package management

### Production Environment
- Docker containerization
- Streamlit Cloud (frontend deployment)
- Cloud hosting for FastAPI backend

### External Services
- OpenWeather API (primary data source)
- AQICN/WAQI API (fallback data source)
- Hopsworks (feature store)
- MLflow (experiment tracking + model registry)
- Evidently AI (monitoring)
- GitHub Actions (CI/CD)

---

## 4. Platform-Specific Notes

### Hopsworks Setup Requirements
- Use `host="eu-west.cloud.hopsworks.ai"` when connecting
- Use `time_travel_format="HUDI"` for feature groups
- Set `online_enabled=True` for online feature serving
- Implement `safe_insert_to_hopsworks()` with retry logic
- Batch writes into single DataFrame operations; avoid loop-based inserts
- Include local fallback when connection limits are exceeded

### Windows Development
- Hopsworks may have issues on Windows locally
- Consider using environment variable toggle (`HOPSWORKS_ENABLED=true/false`)
- GitHub Codespaces can serve as alternative Hopsworks development environment

### Development Environment (Phase 1)
- Repository structure created with 30+ directories
- Python 3.11 virtual environment with pinned dependencies
- Docker: Python 3.11-slim base image
- Code quality: black, isort, flake8 via pre-commit
- Testing: pytest with conftest.py and environment verification tests
- Configuration: config.yaml + .env for secrets
- Hopsworks host loaded from HOPSWORKS_HOST env var (not config.yaml)
- MLflow: local file-based tracking by default (no server required)
- TensorFlow: tensorflow-cpu preferred (lighter install)

---

## 5. Important References

| Document | Purpose |
|---|---|
| MASTER_AGENT_INSTRUCTIONS.md | Highest authority for all project rules |
| source-documents/issues_with_the_workflow.md | Known technical issues and solutions |
| source-documents/hops_work_setup_prompt.md | Hopsworks setup best practices |
| source-documents/answers to students questions from hafsa.md | Mentor Q&A guidance |

---

## 6. Mentor Guidance Notes

- XGBoost with hyperparameter tuning is acceptable if it outperforms Random Forest
- Serverless architecture mainly applies to data/ML pipelines, not frontend/backend
- 72-hour R² of 0.7 is unrealistic; aim for highest achievable
- Model comparison should be documented in write-up report
- Local fallback for Hopsworks is a valid and recommended approach
