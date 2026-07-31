# Memory

## AQI Predictor — Long-Term Project Knowledge

**Version:** 1.0  
**Date:** 31 July 2026  
**Status:** Phase 0 — Foundation  

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

---

## 2. Lessons Learned

*(To be populated as development progresses)*

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
