# Project Journey

## AQI Predictor — Complete Engineering Journey

**Project:** Production-grade AQI forecasting system for Pakistani cities
**Duration:** July–August 2026
**Status:** ✅ Model training complete — XGBoost selected as production model

---

## 1. Initial Problem

Build a machine learning system that predicts Air Quality Index (AQI) at 24, 48, and 72-hour horizons for three Pakistani cities: Karachi, Lahore, and Islamabad.

**Core challenge:** Real-time AQI data from ground monitoring stations in Pakistan is unreliable. AQICN stations for all three cities returned severely stale data — sometimes months or years old. Without fresh training data, no model can be trained.

---

## 2. Architecture Decisions (Phases 0–16)

The project followed a phased development approach:

| Phase | Focus | Key Decision |
|-------|-------|-------------|
| 0 | Foundation | Python 3.11, FastAPI, EPA AQI standard |
| 1–2 | Data collection | OpenWeather primary, AQICN fallback |
| 3–4 | Feature engineering | Lag, rolling, time, derived features |
| 5–6 | Feature store | Hopsworks primary, DuckDB fallback |
| 7–8 | ML pipeline | Ridge, RF, XGBoost, LSTM comparison |
| 9–12 | MLOps | MLflow tracking, lifecycle, CI/CD, API |
| 13–14 | Deployment | Streamlit dashboard, Docker Compose |
| 15–16 | Documentation | Final docs, demo preparation |
| 17 | Data acquisition | Historical dataset generation |

---

## 3. API Selection Journey

### 3.1 OpenWeather (Phase 0–3)

**Initial choice:** OpenWeather API for weather + air pollution data.

- ✅ Weather: temperature, humidity, wind, pressure — worked well
- ✅ Air pollution: PM2.5, PM10, CO, NO2, SO2, O3 — hourly, real-time
- ⚠️ Historical weather: requires paid subscription for >7 days
- ⚠️ AQI scale: OpenWeather returns 1-5 index, NOT US EPA 0-500

**Impact:** Could get real-time data but not historical for training.

### 3.2 AQICN (Phase 0–3)

**Added as fallback** for AQI/pollution data.

- ✅ Provides US EPA AQI (0-500 scale)
- ❌ Pakistani stations severely stale (data from months/years ago)
- ❌ City-level feeds returned static cached values
- ⚠️ Bound station IDs provided fresher data but still 6-7 hours old

**Impact:** AQICN unusable for training data in Pakistan.

### 3.3 AQI Calculation Workaround (Phase 17)

**Problem:** Neither OpenWeather (1-5 scale) nor AQICN (stale) could provide US EPA AQI for training.

**Solution:** Derive US EPA AQI from OpenWeather pollutant concentrations using EPA methodology:
- PM2.5 concentration → EPA breakpoints → AQI sub-index
- PM10 concentration → EPA breakpoints → AQI sub-index
- Select maximum as overall AQI

**Implemented:** EPA-454/B-24-002, May 2024 breakpoints with NowCast algorithm.

**Limitation:** Required 12-hour pollutant history for NowCast; 30-day live collection needed.

### 3.4 Open-Meteo Discovery (Phase 17 Revision)

**Key finding:** Open-Meteo provides free, no-API-key historical data:
- `/v1/archive`: Hourly weather from 2017+ (IFS 9km reanalysis)
- `/v1/air-quality`: Hourly pollutants from Aug 2022+ (CAMS Global 45km)

**This eliminated the 30-day wait entirely.** Instead of collecting data in real-time, we could download 4+ years of historical data in minutes.

---

## 4. Open-Meteo Implementation

### 4.1 Provider Architecture

```
src/data/providers/
├── base_provider.py               — Abstract interface
├── open_meteo_weather.py          — /v1/archive (weather)
└── open_meteo_air_quality.py      — /v1/air-quality (pollutants)
```

**Design principles:**
- Providers are independent of existing OpenWeather/AQICN clients
- No API key required — Open-Meteo is free for non-commercial use
- Chunked downloading respects API limits (92 days per AQ request)
- Rate limiting prevents overload

### 4.2 Ingestion Pipeline

```
historical_ingestion.py:
  1. Download weather (all 3 cities, 2022-08-01 to 2026-08-26)
  2. Download air quality (same range)
  3. Merge on (timestamp, location_id)
  4. Calculate EPA AQI from PM2.5 + PM10
  5. Validate data quality
  6. Save to CSV
```

### 4.3 Dataset Generation Result

| Metric | Value |
|--------|-------|
| Total API requests | 66 |
| API errors | 0 |
| Total download time | ~80 seconds |
| Total rows | 107,064 |
| Cities | 3 |
| Date range | Aug 2022 – Aug 2026 |
| AQI valid rows | 106,848 (99.8%) |

---

## 5. Challenges and Solutions

### 5.1 AQICN Staleness

**Challenge:** AQICN Pakistan stations returned data from months/years ago. City-level feeds were fundamentally stale.

**Solution:** 
- Switched to bound station IDs (temporarily)
- Discovered AQI could be derived from OpenWeather pollutants
- Eventually replaced with Open-Meteo historical data

### 5.2 No Historical Weather on Free Tier

**Challenge:** OpenWeather free tier only provides current weather. Historical requires paid subscription.

**Solution:** Used Open-Meteo Archive API, which provides 80+ years of reanalysis data for free.

### 5.3 AQI Scale Mismatch

**Challenge:** OpenWeather returns AQI on 1-5 scale. Project requires US EPA 0-500 scale.

**Solution:** Implemented EPA AQI calculation from raw pollutant concentrations (PM2.5, PM10). Used official May 2024 breakpoints.

### 5.4 NowCast Algorithm Complexity

**Challenge:** EPA NowCast requires 12-hour pollutant history. Live collection had to accumulate history before producing valid AQI.

**Solution:** Collected 7-day pollution warm-up, then used NowCast history manager to maintain rolling state.

### 5.5 Data Source Migration

**Challenge:** Timeline required immediate historical dataset, not 30-day live collection.

**Solution:** Migrated to Open-Meteo, which provides both weather and air quality historically. Generated 4-year dataset in 2 minutes.

---

## 6. Key Technical Decisions

| ID | Decision | Rationale |
|----|----------|-----------|
| DEC-001 | Python 3.11 | Hopsworks compatibility |
| DEC-002 | FastAPI | Auto docs, async, Pydantic |
| DEC-003 | OpenWeather primary | Dynamic, frequent updates |
| DEC-004 | Hopsworks + DuckDB fallback | Production + local reliability |
| DEC-005 | Multi-output model | Simpler deployment |
| DEC-006 | US EPA AQI | International standard |
| DEC-014 | Data source authority | Clear ownership rules |
| DEC-015 | Synthetic data restricted | Training integrity |
| DEC-018 | Open-Meteo historical | Free, no API key, immediate data |

---

## 7. Dataset Quality Summary

### 7.1 Strengths

- **Large volume:** 107K rows across 4 years
- **Complete weather data:** 0% missing (reanalysis is gap-free)
- **High AQI validity:** 99.8% of rows have valid AQI targets
- **Balanced cities:** Equal representation (35,688 rows each)
- **Diverse pollution profiles:** Lahore (high), Karachi (moderate), Islamabad (lower)
- **Seasonal coverage:** Full annual cycles captured
- **No data leakage:** Verified by construction

### 7.2 Known Issues

- **CAMS AQ resolution:** 45km grid, not ground-station measurements
- **O3 negative values:** 91 rows with negative O3 (CAMS model artifact)
- **216 missing pollution rows:** 9 days of CAMS gaps per city
- **Derived AQI:** Not official EPA monitor readings

### 7.3 Training Readiness

| Criterion | Status |
|-----------|--------|
| Minimum 500 usable rows/city | ✅ (35,616/city) |
| Real API data (not synthetic) | ✅ |
| Valid AQI targets | ✅ (99.8%) |
| No data leakage | ✅ |
| Chronological split | ✅ |
| EPA methodology documented | ✅ |

**Dataset is approved for model training.**

---

## 8. Lessons Learned

1. **Don't assume API availability:** AQICN appeared to work but returned stale data. Always validate data freshness, not just HTTP 200.

2. **Provider abstraction pays off:** Adding Open-Meteo was straightforward because the architecture already had provider abstractions.

3. **Free APIs can be powerful:** Open-Meteo provides production-quality historical data without any API key or subscription.

4. **AQI calculation is complex:** The EPA AQI methodology has specific truncation rules, breakpoint tables, and rounding requirements. Getting it right requires careful implementation.

5. **Data quality validation catches real issues:** Negative O3 values, missing data windows, and staleness detection all found genuine problems.

6. **Chronological splits are essential:** Random splits would leak future information into training, producing misleadingly good metrics.

---

## 9. Model Training Complete

### 9.1 Dataset Generated

| Metric | Value |
|--------|-------|
| Total rows | 107,064 |
| Cities | 3 (Karachi, Lahore, Islamabad) |
| Date range | Aug 2022 – Aug 2026 (4 years) |
| Features | 79 (time, lag, rolling, derived, current) |
| AQI valid | 99.8% |
| Train/Val/Test | 63,648 / 26,280 / 17,136 |
| API calls | 66 (0 errors) |
| Generation time | ~128 seconds |

### 9.2 Documentation Created

| Document | Content |
|----------|---------|
| DATASET_REPORT.md | Full analysis: features, AQI distribution, quality checks |
| PROJECT_JOURNEY.md | This document: engineering history |
| MODEL_EXPERIMENT_PLAN.md | Training strategy, model candidates, selection criteria |

### 9.3 Full Dataset Training Results

**Test set:** 2026 data (16,920 rows) — unseen during training.
**Split:** Train 2022–2024 → Val 2025 → Test 2026.

| Model | MAE | RMSE | R² | Train Time | Inference |
|-------|-----|------|----|-----------|----------|
| **XGBoost** | **21.32** | 30.89 | 0.6065 | 18.2s | 0.030ms |
| RandomForest | 21.47 | **30.74** | **0.6103** | 477.7s | 0.047ms |
| Ridge | 21.98 | 31.99 | 0.5779 | 1.9s | 0.001ms |
| LSTM | 26.17 | 38.86 | 0.3771 | 224.3s | 0.371ms |

### 9.4 Selection Reasoning

**XGBoost selected as production model:**
- Lowest MAE overall (21.32) and at every horizon
- Fastest non-linear training (18.2s vs RF's 477.7s)
- Nearly identical to RF but 26× faster
- Inference speed adequate for real-time API (0.030ms/sample)
- Strong R² (0.6065) — explains 60.65% of AQI variance

**Ridge as backup:** Within 3% of XGBoost — problem has strong linear signal.

**LSTM excluded:** R²=0.3771 vs XGBoost's 0.6065 — temporal patterns already captured by engineered features.

### 9.5 Notebooks Created

| Notebook | Purpose |
|----------|---------|
| 01_dataset_exploration.ipynb | Dataset overview, distributions, patterns |
| 02_feature_analysis.ipynb | Correlations, feature importance, engineering |
| 03_model_experiments.ipynb | Model training and evaluation (all 4 models) |
| 04_model_comparison.ipynb | Final comparison and selection |

---

## 10. Next Steps

1. **Register XGBoost** in MLflow model registry
2. **Deploy** with FastAPI backend + Streamlit dashboard
3. **Set up live monitoring** on production infrastructure
4. **Tune LSTM** on server with GPU when time permits
