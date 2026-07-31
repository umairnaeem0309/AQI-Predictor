# Product Requirements Document (PRD)

## AQI Predictor

**Version:** 1.0  
**Date:** 31 July 2026  
**Status:** Phase 0 — Foundation  

---

## 1. Problem Statement

Air quality significantly impacts public health, particularly in densely populated urban areas. Citizens need timely, accurate forecasts of air quality conditions to make informed decisions about outdoor activities, commuting, and health precautions.

Current solutions often provide only real-time readings without meaningful forward-looking predictions. This project addresses the need for a **3-day AQI forecasting system** that combines automated data collection, machine learning, and an accessible user interface.

---

## 2. Users

| User Type | Description |
|---|---|
| **Health-conscious citizens** | Individuals who need AQI forecasts to plan daily activities |
| **Outdoor workers** | People whose occupations require outdoor exposure and need advance warnings |
| **City residents** | General population in supported cities seeking air quality information |
| **Administrators** | System operators who monitor pipeline health and model performance |

---

## 3. Project Goals

1. **Forecast AQI** at 24-hour, 48-hour, and 72-hour horizons for user-selected cities
2. **Automate data collection** from real-time weather and air pollution APIs
3. **Engineer ML-ready features** from raw environmental data with strict leakage prevention
4. **Experiment with multiple models** (Ridge, Random Forest, XGBoost, LSTM) and select the best through evidence-based comparison
5. **Deploy a production-quality system** with API backend, interactive dashboard, monitoring, and CI/CD
6. **Maintain reproducibility** through experiment tracking, model versioning, and comprehensive documentation

---

## 4. Functional Requirements

### 4.1 Data Collection
- Collect weather data: temperature, humidity, wind speed, pressure, weather conditions
- Collect air pollution data: AQI, PM2.5, PM10, CO, NO2, SO2, O3
- Primary source: OpenWeather API
- Fallback source: AQICN / WAQI API
- Include: error handling, retry mechanism, logging, stale data detection, duplicate prevention, schema validation

### 4.2 Feature Engineering
- **Time-based features:** hour, day, month, weekday, season
- **Historical features:** previous AQI values, lag features, rolling averages
- **Derived features:** AQI change rate, pollutant ratios, weather interaction features
- All features must prevent data leakage (no future information used)
- Feature logic must be reusable, tested, and documented

### 4.3 Historical Data Backfill
- Run feature pipeline for previous dates to generate training data
- Create comprehensive dataset with features, targets, timestamps, location info
- Validate generated records for consistency

### 4.4 Machine Learning Pipeline
- Load features from feature store
- Split datasets respecting chronological order (time-series split)
- Train and compare: Ridge Regression, Random Forest, XGBoost, LSTM
- Evaluate using MAE, RMSE, R²
- Register models with metadata in MLflow

### 4.5 Prediction System
- Multi-output forecast: `[AQI_24h, AQI_48h, AQI_72h]`
- Prediction flow: User → Dashboard → FastAPI → Model Registry → Feature Store → Prediction → Dashboard Result

### 4.6 Explainability
- SHAP-based feature importance
- Dashboard shows: which features influenced prediction, important pollutants, important weather factors

### 4.7 Alerts
- AQI hazard alerts following US EPA AQI category ranges
- Example: "AQI Level: Very Unhealthy. Recommendation: Avoid outdoor activities."

### 4.8 City Support
- User-selectable cities from dashboard
- Initial cities: Karachi, Lahore, Islamabad
- Architecture must be extensible for adding more cities

---

## 5. Non-Functional Requirements

| Requirement | Standard |
|---|---|
| **Code quality** | Clean structure, modular design, type hints, meaningful naming |
| **Testing** | Unit, integration, and end-to-end tests |
| **Documentation** | 15+ documentation files maintained throughout development |
| **Reproducibility** | Every experiment records random seed, dataset version, features, parameters, metrics |
| **Security** | No hardcoded secrets; environment variables for all credentials |
| **Reliability** | Graceful failure handling with fallbacks at every external dependency |
| **Maintainability** | Another engineer must be able to understand, run, and extend the system |

---

## 6. Success Criteria

The project is considered successful **only** when:

- [ ] A user can access the dashboard, select a city (Karachi/Lahore/Islamabad), and receive AQI predictions for 24h/48h/72h
- [ ] Predictions are generated through the complete deployed ML pipeline (not static or manual)
- [ ] Feature pipeline collects real data from APIs automatically
- [ ] Models are trained experimentally with documented comparison
- [ ] Production model is selected through evidence-based reasoning
- [ ] MLflow tracks all experiments and manages model versions
- [ ] Monitoring detects data drift and prediction anomalies
- [ ] CI/CD automates testing and pipeline execution
- [ ] Documentation covers architecture, decisions, troubleshooting, and deployment

---

## 7. Out of Scope (Phase 0)

- Mobile application
- Multi-language support
- Real-time streaming data (uses periodic batch collection)
- Custom user authentication
- Paid-tier cloud services
- Kubernetes deployment
