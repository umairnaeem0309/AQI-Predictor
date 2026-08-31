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
- **After target generation:** 63,504 usable rows

---

## 5. Data Cleaning

- ✅ 0 duplicate (timestamp, city) pairs
- ✅ <0.2% missing values
- ✅ 0 negative values for PM2.5, PM10, CO, NO2, SO2

---

## 6. Feature Engineering

### Features Created (63+ total)

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
| Ridge Regression | Linear | Baseline, fast, interpretable |
| Random Forest | Ensemble | Non-linear, robust |
| XGBoost | Gradient Boosting | State-of-the-art tabular |
| LSTM | Deep Learning | Sequential patterns |

### Verified Results — 4-Year Dataset

#### Overall Test Set

| Model | MAE | RMSE | R² | Composite Score |
|-------|-----|------|----|-----------------|
| **Ridge** | **26.48** | **34.95** | **0.5722** | **33.91** ★ |
| Random Forest | 27.24 | 35.80 | 0.5510 | 35.11 |
| XGBoost | 28.18 | 37.26 | 0.5136 | 37.04 |

#### Per-Horizon — 24h

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| **Ridge** | **22.50** | **29.72** | **0.6847** ★ |
| Random Forest | 23.29 | 30.85 | 0.6602 |
| XGBoost | 24.43 | 32.40 | 0.6253 |

#### Per-Horizon — 48h

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| **Ridge** | **27.21** | **35.64** | **0.5536** ★ |
| Random Forest | 27.61 | 35.80 | 0.5497 |
| XGBoost | 28.16 | 37.12 | 0.5156 |

#### Per-Horizon — 72h

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| **Ridge** | **29.72** | **38.86** | **0.4784** ★ |
| Random Forest | 30.82 | 40.16 | 0.4431 |
| XGBoost | 31.94 | 41.69 | 0.3998 |

### Selection Rationale

**Ridge Regression** is selected as the production model because:
1. **Best Test MAE** (26.48) — lowest prediction error across all models
2. **Best Test R²** (0.5722) — explains most variance in AQI
3. **Wins ALL 3 horizons** — consistent 24h, 48h, 72h performance
4. **Fastest inference** (0.000 ms) — production-ready
5. **Most interpretable** — linear coefficients directly explain feature influence
6. **Least overfitting risk** — simple model with regularization
7. **Fastest training** — suitable for daily retraining

### Why Not XGBoost?

While XGBoost is state-of-the-art for many tabular problems, on this specific AQI forecasting task:
- **Higher MAE** (28.18 vs 26.48) — 6.4% worse prediction error
- **Lower R²** (0.5136 vs 0.5722) — explains less variance
- **Consistently worse** across all 3 horizons
- The AQI prediction relationships appear sufficiently linear for Ridge to outperform more complex models

### Why Not Random Forest?

- **Second-best** but still worse than Ridge on all metrics
- **Slower inference** (0.009 ms vs 0.000 ms)
- More complex model with no performance benefit

---

## 8. Model Registry

- Stored in Hopsworks Model Registry
- All model metrics logged
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
| Historical CSV had fewer rows | Re-fetched full 4-year data from Open-Meteo |

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
- **Ridge selected as production model** (best on 4-year data)
- **Hopsworks Model Registry**
- **Both services deployed and accessible**
