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

- **Weather:** ~2.5 years hourly data (Aug 2022 – Dec 2024)
- **Air Quality:** ~2.5 years hourly data (Aug 2022 – Dec 2024)
- **Cities:** Karachi (24.86°N, 67.00°E), Lahore (31.52°N, 74.36°E), Islamabad (33.68°N, 73.05°E)
- **Total raw rows:** 107,064 (35,688 per city)

**Note on Data Coverage:** The dataset covers approximately 2.5 years, not the originally planned 5 years. Open-Meteo's historical air quality data (CAMS Global) starts from Aug 2022, which is the earliest available for hourly pollutant data. Weather data is available further back, but the model requires both weather and pollution features aligned at the same timestamps.

### Verification

- ✅ Weather API returns 24 hourly records per city per day
- ✅ Air Quality API returns 24 hourly records per city per day
- ✅ Live fetcher returns current weather + pollution for all 3 cities
- ✅ No API key required for Open-Meteo

---

## 5. Data Cleaning

### Cleaning Steps Applied

1. **Timestamp normalization:** All timestamps converted to UTC
2. **Duplicate removal:** 0 duplicate (timestamp, city) pairs found
3. **Missing values:** <0.2% across all features
4. **Negative pollutants:** 0 negative values for PM2.5, PM10, CO, NO2, SO2
5. **AQI calculation:** EPA PM Direct method using PM2.5 and PM10 concentrations

### Verification

- ✅ All 3 cities have equal row counts (35,688 each)
- ✅ Date range: 2022-08-03 to 2024-12-28
- ✅ No synthetic data marked for training
- ✅ Dataset metadata records provenance

---

## 6. EDA

Four Jupyter notebooks were created for Exploratory Data Analysis:

| Notebook | Purpose |
|----------|---------|
| `01_dataset_exploration.ipynb` | Dataset overview, AQI distribution, city comparison, missing values |
| `02_feature_analysis.ipynb` | Correlation analysis, feature importance preparation |
| `03_model_experiments.ipynb` | Model training experiments, metrics comparison |
| `04_model_comparison.ipynb` | Final comparison, visualizations, selection reasoning |

---

## 7. Feature Engineering

### Features Created (68 total)

| Category | Count | Examples |
|----------|-------|----------|
| Weather | 7 | temperature, humidity, pressure, wind_speed, wind_direction, cloud_cover, precipitation |
| Pollution | 6 | pm25, pm10, co, no2, so2, o3 |
| Time | 6 | hour, day_of_week, month, is_weekend, hour_sin, hour_cos |
| Lag | 24 | aqi_lag_{1,6,12,24,48,72}h, pm25_lag_{1,6,12,24,48,72}h, temperature/humidity lags |
| Rolling | 10 | aqi_rolling_{mean,std,min,max}_{6h,12h,24h}, pm25/temperature/humidity rolling |
| Ratio | 3 | pm25_pm10_ratio, no2_so2_ratio, o3_no2_ratio |
| Change/Trend | 5 | aqi_change_rate_{1h,6h,24h}, aqi_trend_24h, aqi_deviation_from_24h_avg |
| Derived | 3 | temp_humidity_interaction, wind_cooling_effect, aqi_deviation |

### Verification

- ✅ All 68 features present in processed training data
- ✅ hour_sin range: [-1.0, 1.0] (correct)
- ✅ is_weekend: [0, 1] (correct)
- ✅ Feature engineering functions execute without errors
- ✅ No data leakage: targets are future-shifted correctly

---

## 8. Feature Store Setup

### Hopsworks Integration

1. Created `src/feature_store/hopsworks_store.py`
2. Uploaded 63,648 rows to `aqi_features_prod` v1
3. Verified read-back: 63,648 rows, 73 columns
4. Local Parquet fallback implemented

### Verification

- ✅ Hopsworks connected: `eu-west.cloud.hopsworks.ai`
- ✅ Feature group `aqi_features_prod` exists with data
- ✅ Read-back returns correct schema and values
- ✅ Fallback to local Parquet works

---

## 9. Model Training & Selection

### Models Trained

| Model | Type | Why Selected |
|-------|------|--------------|
| Ridge Regression | Linear | Baseline, fast, interpretable |
| Random Forest | Ensemble | Non-linear, robust |
| XGBoost | Gradient Boosting | State-of-the-art tabular |
| LSTM | Deep Learning | Sequential pattern capture |

### Verified Results — Complete Dataset

#### Overall Comparison (Validation Set)

| Model | MAE | RMSE | R² | Composite Score | Train Time |
|-------|-----|------|----|-----------------|------------|
| **Random Forest** | **18.80** | **26.08** | **0.3013** | **36.31** | 190.8s |
| Ridge Regression | 17.95 | 26.30 | 0.2894 | 36.39 | 1.4s |
| LSTM | 20.03 | 26.87 | 0.2582 | 38.33 | 144.5s |
| XGBoost | 20.35 | 26.84 | 0.2597 | 38.40 | 24.7s |

#### Overall Comparison (Test Set)

| Model | MAE | RMSE | R² | Inference Latency |
|-------|-----|------|----|-------------------|
| **Random Forest** | **22.59** | **30.37** | **0.6281** | 0.048 ms/sample |
| Ridge Regression | 23.49 | 31.20 | 0.6077 | 0.001 ms/sample |
| XGBoost | 23.45 | 31.38 | 0.6031 | 0.012 ms/sample |
| LSTM | 23.97 | 31.95 | 0.5882 | 0.159 ms/sample |

#### Per-Horizon Test Set Results

| Horizon | Model | MAE | RMSE | R² |
|---------|-------|-----|------|----|
| **24h** | Ridge | 19.68 | 26.47 | **0.7117** |
| **24h** | Random Forest | **19.39** | 26.93 | 0.7016 |
| **24h** | XGBoost | 20.04 | 27.10 | 0.6978 |
| **24h** | LSTM | 21.60 | 28.97 | 0.6546 |
| **48h** | Ridge | 24.34 | 31.94 | 0.5851 |
| **48h** | Random Forest | **23.36** | **30.83** | **0.6135** |
| **48h** | XGBoost | 24.16 | 31.71 | 0.5912 |
| **48h** | LSTM | 24.37 | 32.38 | 0.5738 |
| **72h** | Ridge | 26.46 | 34.64 | 0.5262 |
| **72h** | Random Forest | **25.00** | **33.03** | **0.5692** |
| **72h** | XGBoost | 26.16 | 34.85 | 0.5203 |
| **72h** | LSTM | 25.94 | 34.28 | 0.5361 |

#### Best Model Per Horizon (Validation Composite)

| Horizon | Best Model | Composite Score |
|---------|------------|----------------|
| 24h | Random Forest | 28.98 |
| 48h | Ridge | 37.52 |
| 72h | Random Forest | 41.13 |

### Selection Rationale

**Random Forest** is selected as the production model because:

1. **Lowest composite score** on validation (36.31 vs Ridge 36.39) — this considers MAE (40%), RMSE (30%), and R² (30%) together across all horizons
2. **Best test performance** across all metrics: MAE=22.59, RMSE=30.37, R²=0.6281
3. **Wins on 2 out of 3 horizons** on test set (24h MAE and 72h MAE)
4. **Most consistent** — never the worst performer on any horizon
5. **Fast inference** — 0.048ms/sample, well within production latency requirements

**Why not Ridge?** Ridge has the lowest validation MAE (17.95) but loses on RMSE and R². The composite score captures this tradeoff — Ridge's composite (36.39) is slightly worse than RF's (36.31).

**Why not XGBoost?** XGBoost was expected to outperform but scored highest composite (38.40). With 2.5 years of data and 68 features, XGBoost may be overfitting on validation.

**Why not LSTM?** LSTM requires sequential data reshaping and scored 38.33 composite. The single-timestep input (reshaped to [samples, 1, features]) limits LSTM's ability to capture temporal patterns that multi-step sequences would provide.

---

## 10. Model Registry

- Registered in MLflow experiment `aqi_predictor_production`
- All 4 model metrics logged (MAE, RMSE, R² per horizon)
- Model artifacts saved locally: `models/production/best_model.pkl`
- API loads model via MLflow → pickle fallback

---

## 11. Deployment

| Component | Platform | URL |
|-----------|----------|-----|
| API Backend | Render | https://aqi-predictor-api-nf7s.onrender.com |
| Dashboard | Streamlit Cloud | https://airpulse.streamlit.app/ |

Both auto-deploy on git push to `main`.

---

## 12. Automation & Monitoring

### GitHub Actions

| Workflow | Schedule | Action |
|----------|----------|--------|
| `feature-collection.yml` | Every hour | Collect weather + pollution, store in Hopsworks |
| `daily-training.yml` | Daily 6 AM UTC | Train all models, select best, register in MLflow |
| `ci.yml` | On push | Lint, format, type-check, unit tests |
| `ml-validation.yml` | Weekly | Data safety, feature quality, model artifact validation |
| `cd.yml` | On push | Pre-deployment checks, Docker build |

### Monitoring

- Evidently AI for data drift detection
- AQI hazard alerts (threshold-based)
- System health endpoint

---

## 13. Blockers & Solutions

| Blocker | Solution |
|---------|----------|
| AQICN Pakistan stations stale | Switched to Open-Meteo |
| OpenWeather free tier limitations | Selected Open-Meteo (no key required) |
| Feature engineering NaN errors | Implemented proper NaN handling with `np.nan_to_num` |
| Hopsworks connection timeout | Added retry logic and local Parquet fallback |
| Model saved as wrong type | Fixed train_model.py to save correct best model |
| CI validation scripts gitignored | Added `!scripts/validate_production.py` to .gitignore |
| Pre-deploy checks require API_KEY | Changed to warning in CI environments |
| SHAP failing for Ridge model | Added LinearExplainer fallback for non-tree models |
| Data routes reading only CSV | Updated to use Hopsworks as primary data source |
| Model selection based on single metric | Changed to composite score (MAE+RMSE+R²) across horizons |

---

## 14. Test Results

```
487 passed, 1 skipped, 0 failed
```

The 1 skipped test is a conditional skip when the master CSV file is not present — legitimate behavior.

---

## 15. Current System State

- **All pipelines verified end-to-end**
- **487 tests passing**
- **Hopsworks connected with 63,648 rows**
- **Random Forest selected via composite scoring**
- **MLflow tracking experiments**
- **CI/CD all green**
- **Both services deployed and accessible**
