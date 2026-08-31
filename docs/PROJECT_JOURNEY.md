# AQI Predictor — Project Journey

**Last Updated:** 2026-08-31

---

## 1. Problem Definition

Build a production-grade AQI forecasting system that predicts Air Quality Index 24, 48, and 72 hours ahead for three Pakistani cities: Karachi, Lahore, and Islamabad.

---

## 2. API Selection Process

### Initial Attempt: OpenWeather + AQICN

We initially planned to use:
- **OpenWeather API** for weather data
- **AQICN API** for US EPA AQI data

**Problem discovered:** AQICN Pakistan stations were returning stale data (months/years old). The returned timezone offsets (-05:00, -06:00) were incorrect for Pakistani cities, causing timestamp normalization bugs.

**Decision:** Rejected AQICN as primary AQI source.

### OpenWeather Investigation

OpenWeather had a separate Air Pollution API with historical data. However:
- Free tier limited historical access
- PM2.5 required 24-hour averaging for correct EPA AQI calculation
- Complex gas concentration unit conversions needed

**Decision:** OpenWeather was rejected due to API limitations on the free tier.

### Final Choice: Open-Meteo

Open-Meteo provided:
- **No API key required** for non-commercial use
- **Historical weather data** from 2017+ (IFS 9km) or 1940+ (ERA5)
- **Historical air quality data** from Aug 2022+ (CAMS Global)
- **Hourly granularity** for all pollutants
- **Free tier** with generous rate limits

**Decision:** Open-Meteo selected as the primary data provider.

---

## 3. Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Weather API | Open-Meteo Archive | Free, historical, hourly |
| Air Quality API | Open-Meteo Air Quality | Free, PM2.5/PM10/CO/NO2/SO2/O3 |
| Feature Store | Hopsworks (PRIMARY) | Cloud-based, versioned, scalable |
| Fallback Store | Local Parquet | Works offline, no cloud dependency |
| Model Registry | MLflow (local) | Version tracking, metrics logging |
| Web Framework | FastAPI + Streamlit | Async API + interactive dashboard |
| CI/CD | GitHub Actions | Automated hourly + daily pipelines |
| Deployment | Render (API) + Streamlit Cloud (Dashboard) | Free tier, auto-deploy |

---

## 4. Data Collection

### Historical Data Downloaded

- **Weather:** ~4 years hourly data (Aug 2022 – Aug 2026)
- **Air Quality:** ~4 years hourly data (Aug 2022 – Aug 2026)
- **Cities:** Karachi (24.86°N, 67.00°E), Lahore (31.52°N, 74.36°E), Islamabad (33.68°N, 73.05°E)
- **Total rows:** 107,208 (35,736 per city)

**Note:** Open-Meteo's historical air quality data (CAMS Global) starts from Aug 2022. We have the maximum available range.

---

## 5. Data Cleaning

- ✅ All 3 cities have equal row counts (35,736 each)
- ✅ Date range: 2022-08-04 to 2026-08-28
- ✅ 0 duplicate (timestamp, city) pairs
- ✅ <0.2% missing values
- ✅ 0 negative values for PM2.5, PM10, CO, NO2, SO2
- ✅ 91 negative O3 values (investigated — within acceptable range)

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
| AQI/Ratio | 10 |

---

## 7. Model Training & Selection

### Models Trained

| Model | Type | Why Selected |
|-------|------|--------------|
| Ridge Regression | Linear | Baseline, fast, interpretable |
| Random Forest | Ensemble | Non-linear, robust |
| XGBoost | Gradient Boosting | State-of-the-art tabular |
| LSTM | Deep Learning | Sequential pattern capture |

### Verified Results — 4-Year Dataset

#### Overall Comparison (Validation Set)

| Model | MAE | RMSE | R² | Composite Score | Train Time |
|-------|-----|------|----|-----------------|------------|
| **Random Forest** | **19.18** | **26.84** | **0.5019** | **30.67** | 178.9s |
| XGBoost | 19.41 | 27.36 | 0.4826 | 31.50 | 16.9s |
| Ridge | 19.62 | 27.37 | 0.4822 | 31.59 | 0.8s |
| LSTM | 19.97 | 27.81 | 0.4654 | 32.37 | 89.9s |

#### Overall Comparison (Test Set)

| Model | MAE | RMSE | R² | Inference Latency |
|-------|-----|------|----|-------------------|
| **XGBoost** | **21.34** | **30.35** | **0.6584** | 0.011 ms/sample |
| Random Forest | 21.61 | 30.58 | 0.6533 | 0.013 ms/sample |
| Ridge | 21.73 | 30.64 | 0.6520 | 0.0003 ms/sample |
| LSTM | 22.95 | 32.46 | 0.6092 | 0.057 ms/sample |

#### Per-Horizon Test Set Results

| Horizon | Best Model | MAE | RMSE | R² |
|---------|------------|-----|------|----|
| **24h** | XGBoost | **19.00** | **27.43** | **0.7206** |
| **48h** | XGBoost | **21.81** | **30.89** | **0.6461** |
| **72h** | XGBoost | **23.23** | **32.51** | **0.6085** |

#### Best Model Per Horizon (Validation Composite)

| Horizon | Best Model | Composite Score |
|---------|------------|----------------|
| 24h | XGBoost | 26.56 |
| 48h | Random Forest | 32.09 |
| 72h | Random Forest | 33.28 |

### Selection Rationale

**Random Forest** is selected as the production model because:
1. **Lowest validation composite** (30.67 vs XGBoost 31.50)
2. **Prevents overfitting** — validation score better predicts generalization
3. **Consistent performance** — never the worst on any horizon
4. **Fast inference** (0.013 ms/sample)

**Honest assessment:** XGBoost performs slightly better on the test set (MAE 21.34 vs 21.61, R² 0.6584 vs 0.6533). The difference is small. RandomForest was selected based on validation composite to avoid test-set overfitting.

---

## 8. Model Registry

- Registered in MLflow experiment `aqi_predictor_production`
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
| `cd.yml` | On push | Pre-deploy checks, Docker |

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
- **Random Forest selected via composite scoring**
- **MLflow tracking experiments**
- **Both services deployed and accessible**
