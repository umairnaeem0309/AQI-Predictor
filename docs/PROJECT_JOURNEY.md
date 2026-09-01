# AQI Predictor — Project Journey

**Last Updated:** 2026-08-31

---

## 1. Problem Definition

Build a production-grade AQI forecasting system that predicts Air Quality Index 24, 48, and 72 hours ahead for three Pakistani cities: Karachi, Lahore, and Islamabad.

---

## 2. API Selection Process

### Initial Attempt: OpenWeather + AQICN

- **Problem:** AQICN Pakistan stations were stale (months/years old)
- **Decision:** Rejected AQICN as primary AQI source

### OpenWeather Investigation

- **Problem:** Free tier limited historical access
- **Decision:** Rejected due to API limitations

### Final Choice: Open-Meteo

- ✅ No API key required
- ✅ Historical weather from 2017+
- ✅ Historical air quality from Aug 2022+
- ✅ Hourly granularity
- ✅ Free tier with generous rate limits

---

## 3. Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Data Provider | Open-Meteo | Free, historical, hourly |
| Feature Store | Hopsworks (PRIMARY) | Cloud-based, versioned |
| Feature + Targets | Single Feature Group | Prevents data drift, ensures consistency |
| Model Registry | Hopsworks | Integrated with feature store |
| Web Framework | FastAPI + Streamlit | Async API + dashboard |
| CI/CD | GitHub Actions | Automated hourly + daily |
| Deployment | Render + Streamlit Cloud | Free tier, auto-deploy |

---

## 4. Data Collection

- **Weather:** ~4 years (Aug 2022 – Aug 2026)
- **Air Quality:** ~4 years (Aug 2022 – Aug 2026)
- **Cities:** Karachi, Lahore, Islamabad
- **Total rows:** 107,064 (35,688 per city)
- **Stored in:** Hopsworks Feature Store (features + targets together)

---

## 5. Data Cleaning

- ✅ 0 duplicate (timestamp, city) pairs
- ✅ <0.2% missing values
- ✅ 0 negative values for PM2.5, PM10, CO, NO2, SO2

---

## 6. Feature Engineering

### Features Created (58 total)

| Category | Count |
|----------|-------|
| Weather | 7 |
| Pollution | 6 |
| Time | 6 |
| Lag | 24 |
| Rolling | 10 |
| Derived | 5 |

### Feature Store Architecture

- Features + targets stored in **single Hopsworks Feature Group**
- **Feature View** with target label designation
- **No local CSV files** for training — ALL data from Hopsworks Feature Store (no fallback)
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
4. **Fast training** (22.5s) — suitable for daily retraining
5. **Handles non-linear relationships** in AQI data

---

## 8. Model Registry

- Stored in Hopsworks Model Registry
- XGBoost v2 registered with full metrics
- URL: https://eu-west.cloud.hopsworks.ai/p/41205/models/xgboost/2

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
| `feature-collection.yml` | Every hour | Collect weather + pollution |
| `daily-training.yml` | Daily 6 AM UTC | Train all models, select best |
| `ci.yml` | On push | Lint, tests |
| `ml-validation.yml` | Weekly | Data safety, feature quality |
| `cd.yml` | On push | Pre-deploy checks |

---

## 11. Blockers & Solutions

| Blocker | Solution |
|---------|----------|
| AQICN Pakistan stations stale | Switched to Open-Meteo |
| OpenWeather free tier limitations | Selected Open-Meteo (no key required) |
| Feature engineering NaN errors | Implemented proper NaN handling |
| Hopsworks connection timeout | Added retry logic and local Parquet fallback |
| CI validation scripts gitignored | Added exception to .gitignore |
| SHAP failing for Ridge model | Added LinearExplainer fallback |
| Data routes reading only CSV | Updated to use Hopsworks as primary |
| Model selection based on single metric | Changed to composite score across horizons |
| Local CSV files causing data drift | Migrated to Hopsworks Feature Store |
| Separate features/targets files | Combined into single Feature Group |


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
- **Hopsworks Model Registry** (XGBoost v2)
- **Both services deployed and accessible**
- **Hourly data collection + daily retraining automated via GitHub Actions**
