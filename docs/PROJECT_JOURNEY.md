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
| Model Registry | Hopsworks | Integrated with feature store |
| Web Framework | FastAPI + Streamlit | Async API + dashboard |
| CI/CD | GitHub Actions | Automated hourly + daily |
| Deployment | Render + Streamlit Cloud | Free tier, auto-deploy |

---

## 4. Data Collection

- **Weather:** ~4 years (Aug 2022 – Aug 2026)
- **Air Quality:** ~4 years (Aug 2022 – Aug 2026)
- **Cities:** Karachi, Lahore, Islamabad
- **Total rows:** 107,208 (35,736 per city)

---

## 5. Data Cleaning

- ✅ 0 duplicate (timestamp, city) pairs
- ✅ <0.2% missing values
- ✅ 0 negative values for PM2.5, PM10, CO, NO2, SO2

---

## 6. Feature Engineering

### Features Created (63 total)

| Category | Count |
|----------|-------|
| Weather | 7 |
| Pollution | 6 |
| Time | 6 |
| Lag | 24 |
| Rolling | 10 |
| Derived | 10 |

---

## 7. Model Training & Selection

### Models Trained

| Model | Type | Why Selected |
|-------|------|--------------|
| Ridge Regression | Linear | Baseline, fast |
| Random Forest | Ensemble | Non-linear, robust |
| XGBoost | Gradient Boosting | State-of-the-art tabular |
| LSTM | Deep Learning | Sequential patterns |

### Verified Results — 4-Year Dataset

#### Test Set

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| **XGBoost** | **21.34** | **30.35** | **0.6584** |
| Random Forest | 21.61 | 30.58 | 0.6533 |
| Ridge | 21.73 | 30.64 | 0.6520 |
| LSTM | 22.95 | 32.46 | 0.6092 |

#### Per-Horizon Test Set

| Horizon | Best Model | MAE | R² |
|---------|------------|-----|----|
| 24h | XGBoost | 19.00 | 0.7206 |
| 48h | XGBoost | 21.81 | 0.6461 |
| 72h | XGBoost | 23.23 | 0.6085 |

### Selection Rationale

**XGBoost** is selected as the production model because:
1. **Best Test MAE** (21.34) — lowest prediction error
2. **Best Test R²** (0.6584) — explains most variance
3. **Wins ALL 3 horizons** — consistent performance
4. **Fast training** (9.9s) — suitable for daily retraining
5. **Fast inference** (0.011 ms) — production-ready

---

## 8. Model Registry

- Stored in Hopsworks Model Registry
- All 4 model metrics logged
- Model artifacts saved locally: `models/production/best_model.pkl`

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

---

## 12. Test Results

```
487 passed, 1 skipped, 0 failed
```

---

## 13. Current System State

- **All pipelines verified end-to-end**
- **487 tests passing**
- **Hopsworks connected with 107,208 rows**
- **4-year dataset (2022-08 to 2026-08)**
- **XGBoost selected as production model**
- **Hopsworks Model Registry**
- **Both services deployed and accessible**
