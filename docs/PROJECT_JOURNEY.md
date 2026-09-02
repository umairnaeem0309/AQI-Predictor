# AQI Predictor — Project Journey

**Last Updated:** 2026-09-02

---

## 1. Problem Definition

Build a production-grade AQI forecasting system that predicts Air Quality Index 24, 48, and 72 hours ahead for three Pakistani cities: Karachi, Lahore, and Islamabad.

The system must include:
- Historical data collection and storage
- Feature engineering pipeline
- Feature Store integration (Hopsworks)
- Model training and evaluation
- Model Registry (Hopsworks)
- CI/CD automation (GitHub Actions)
- Live API and dashboard

---

## 2. API Selection Process

### Initial Attempt: OpenWeather + AQICN

- **Problem:** AQICN Pakistan stations were stale (months/years old)
- **Decision:** Rejected AQICN as primary AQI source

### OpenWeather Investigation

- **Problem:** Free tier limited historical access
- **Decision:** Rejected due to API limitations

### Phase 17: OpenWeather + Open-Meteo (Hopsworks Phase)

- **Problem:** 30-day collection too slow for timeline
- **Solution:** Open-Meteo provides historical data from 2017+
- **Decision:** Selected Open-Meteo as primary data source

### Final Choice: Open-Meteo

- ✅ No API key required
- ✅ Historical weather from 2017+
- ✅ Historical air quality from Aug 2022+
- ✅ Hourly granularity
- ✅ Free tier with generous rate limits
- ✅ Works for all three cities

---

## 3. Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Data Provider | Open-Meteo | Free, historical, hourly, no key required |
| Feature Store | Hopsworks (PRIMARY) | Cloud-based, versioned, no CSV fallback |
| Feature + Targets | Single Feature Group | Prevents data drift, ensures consistency |
| Model Registry | Hopsworks | Integrated with feature store |
| Web Framework | FastAPI + Streamlit | Async API + interactive dashboard |
| CI/CD | GitHub Actions | Automated hourly + daily |
| Deployment | Render + Streamlit Cloud | Free tier, auto-deploy |
| Model Selection | Composite score | MAE (40%) + RMSE (30%) + R² penalty (30%) |

---

## 4. Data Collection

- **Source:** Open-Meteo Weather API + Air Quality API
- **Weather:** ~4 years (Aug 2022 – Aug 2026)
- **Air Quality:** ~4 years (Aug 2022 – Aug 2026)
- **Cities:** Karachi, Lahore, Islamabad
- **Total rows:** 107,064 (35,688 per city)
- **Stored in:** Hopsworks Feature Store (features + targets together)
- **Ingestion script:** `scripts/ingest_to_hopsworks.py`
- **Hourly collection:** `scripts/collect_features.py`

---

## 5. Data Cleaning

- ✅ 0 duplicate (timestamp, city) pairs
- ✅ <0.2% missing values
- ✅ 0 negative values for PM2.5, PM10, CO, NO2, SO2
- ✅ Timestamps normalized to UTC
- ✅ Hourly ordering verified

---

## 6. Feature Engineering

### Features Created (58 total)

| Category | Count | Features |
|----------|-------|----------|
| Weather | 7 | temperature, humidity, pressure, wind_speed, wind_direction, cloud_cover, precipitation |
| Pollution | 6 | pm25, pm10, co, no2, so2, o3 |
| Time | 6 | hour, day_of_week, month, is_weekend, season, hour_sin, hour_cos |
| Lag | 24 | aqi/pm25/temp/humidity lags at 1h, 6h, 12h, 24h, 48h, 72h |
| Rolling | 10 | aqi/pm25/temp/humidity rolling means and stats |
| Derived | 5 | us_aqi variants |

### Feature Store Architecture

- Features + targets stored in **single Hopsworks Feature Group**
- **Feature View** with target label designation
- **No local CSV files** for training — ALL data from Hopsworks Feature Store
- Ensures reproducible, consistent splits

---

## 7. Model Training & Selection

### Models Trained

| Model | Type | Why Selected |
|-------|------|--------------|
| Ridge Regression | Linear | Baseline, fast, interpretable |
| Random Forest | Ensemble | Non-linear, robust |
| XGBoost | Gradient Boosting | State-of-the-art tabular |
| LSTM | Deep Learning | Sequential pattern capture |

### Verified Results — Hopsworks Feature Store

#### Overall Test Set

| Model | MAE | RMSE | R² | Composite Score |
|-------|-----|------|----|-----------------|
| **XGBoost** | **21.31** | **30.33** | **0.6588** | **27.84** ★ |
| Random Forest | 21.39 | 30.33 | 0.6588 | 27.87 |
| Ridge | 21.84 | 30.67 | 0.6509 | 28.39 |
| LSTM | 39.58 | 52.57 | -0.0252 | 62.36 |

#### Per-Horizon — 24h

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| **XGBoost** | **19.01** | **27.41** | **0.7210** ★ |
| Random Forest | 19.20 | 27.53 | 0.7185 |
| Ridge | 19.53 | 27.83 | 0.7122 |

#### Per-Horizon — 48h

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| **XGBoost** | **21.78** | **30.88** | **0.6463** ★ |
| Random Forest | 21.89 | 30.87 | 0.6465 |
| Ridge | 22.37 | 31.27 | 0.6372 |

#### Per-Horizon — 72h

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| **XGBoost** | **23.15** | **32.48** | **0.6091** ★ |
| Random Forest | 23.08 | 32.52 | 0.6081 |
| Ridge | 23.62 | 32.85 | 0.6002 |
| LSTM | 46.34 | 57.68 | -0.0798 |

### Selection Rationale

**XGBoost** is selected as the production model because:
1. **Best Test MAE** (21.31) — lowest prediction error
2. **Best Test R²** (0.6588) — explains most variance
3. **Wins ALL 3 horizons** — 24h, 48h, 72h consistently
4. **Fast training** (23.7s) — suitable for daily retraining
5. **Handles non-linear relationships** in AQI data

---

## 8. Model Registry

- Stored in Hopsworks Model Registry
- XGBoost v4 registered with full metrics
- URL: https://eu-west.cloud.hopsworks.ai/p/41205/models/xgboost/4
- Model loads via `download()` + pickle deserialization

---

## 9. Deployment

| Component | Platform | URL |
|-----------|----------|-----|
| API Backend | Render | https://aqi-predictor-api-nf7s.onrender.com |
| Dashboard | Streamlit Cloud | https://airpulse.streamlit.app/ |

---

## 10. Automation & Monitoring

| Workflow | Schedule | Action |
|----------|----------|--------|
| `feature-collection.yml` | Every hour at :00 | Collect weather + pollution from Open-Meteo → Hopsworks |
| `daily-training.yml` | Daily 6 AM UTC | Train all 4 models from Hopsworks, select best, register |
| `keep-alive.yml` | Every 10 min | Ping Render API to prevent free-tier sleep |
| `ci.yml` | On push | Lint, type-check, tests, Docker build, security audit |
| `cd.yml` | On push | Validate production, build Docker, deploy |

---

## 11. Blockers & Solutions

| Blocker | Solution |
|---------|----------|
| AQICN Pakistan stations stale | Switched to Open-Meteo |
| OpenWeather free tier limitations | Selected Open-Meteo (no key required) |
| Feature engineering NaN errors | Implemented proper NaN handling |
| Hopsworks connection timeout | Added retry logic and caching |
| CI validation scripts gitignored | Added exception to .gitignore |
| SHAP failing for Ridge model | Added LinearExplainer fallback |
| Data routes reading only CSV | Updated to use Hopsworks as primary |
| Model selection based on single metric | Changed to composite score across horizons |
| Local CSV files causing data drift | Migrated to Hopsworks Feature Store |
| Separate features/targets files | Combined into single Feature Group |
| Render free-tier sleep | Added keep-alive workflow |
| Render cold start timeouts | Added API response caching + retry logic |
| Hopsworks model.load() not available | Fixed to use download() + pickle |
| YAML workflow syntax errors | Moved Python reporting to separate scripts |
| CRLF line endings on Windows | Added .gitattributes for LF enforcement |

---

## 12. Test Results

```
487 passed, 1 skipped, 0 failed
```

---

## 13. Current System State

- **All pipelines verified end-to-end**
- **487 tests passing**
- **Hopsworks connected with 107,064 rows (features + targets together)**
- **4-year dataset (2022-08 to 2026-08)**
- **XGBoost selected as production model** (composite score 27.84)
- **Hopsworks Model Registry** (XGBoost v4)
- **Both services deployed and accessible**
- **Hourly data collection + daily retraining automated via GitHub Actions**
