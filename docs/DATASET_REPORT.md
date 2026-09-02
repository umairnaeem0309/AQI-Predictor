# Dataset Report

## AQI Predictor — Historical Dataset Analysis

**Generated:** 31 August 2026
**Status:** Ready for model training
**Dataset type:** real_api_data (approved_for_training: true)

---

## 1. Dataset Overview

| Metric | Value |
|--------|-------|
| Total rows | 107,208 |
| Cities | 3 (Karachi, Lahore, Islamabad) |
| Date range | 2022-08-04 to 2026-08-28 |
| Time span | **~4 years** |
| Rows per city | 35,736 |
| Weather features | 7 |
| Pollution features | 6 |
| Total features | 63 |
| Target variables | 3 (target_aqi_24h, target_aqi_48h, target_aqi_72h) |

---

## 2. Data Sources

| Source | Endpoint | Data | Resolution | History |
|--------|----------|------|-----------|---------|
| Open-Meteo Weather | `/v1/archive` | temperature, humidity, pressure, wind, clouds, precipitation | Hourly | 2017+ (IFS 9km) |
| Open-Meteo Air Quality | `/v1/air-quality` | PM2.5, PM10, CO, NO2, SO2, O3 | Hourly | Aug 2022+ (CAMS Global 45km) |

---

## 3. AQI Distribution

### Overall

| Statistic | Value |
|-----------|-------|
| Valid AQI values | 107,208 (100%) |
| Range | 2 – 500 |
| Mean | 114.7 |
| Median | 100.0 |

### Per-City

| City | n | Mean | Median | Min | Max |
|------|---|------|--------|-----|-----|
| Islamabad | 35,736 | 105.6 | 97 | 3 | 343 |
| Karachi | 35,736 | 90.7 | 81 | 2 | 471 |
| Lahore | 35,736 | 147.8 | 148 | 3 | 500 |

---

## 4. Train/Val/Test Split

| Split | Rows | Percentage |
|-------|------|-----------|
| Train (72%) | 77,034 | 72% |
| Validation (8%) | 8,559 | 8% |
| Test (20%) | 21,399 | 20% |

**Split method:** Chronological (sorted by timestamp).

---

## 5. Data Quality

- **Duplicates:** 0
- **Missing values:** <0.2%
- **Negative pollutants:** 0 for PM2.5, PM10, CO, NO2, SO2 (91 negative O3 from CAMS model)

---

## 6. Feature Engineering

| Category | Count |
|----------|-------|
| Weather | 7 |
| Pollution | 6 |
| Time | 6 |
| Lag | 24 |
| Rolling | 10 |
| Derived | 10 |
| **Total** | **63** |

---

## 7. Reproducibility

```bash
# Fetch 4-year data
python scripts/backfill_full_4years.py

# Feature engineering + upload to Hopsworks
python scripts/backfill_hopsworks_full.py

# Train models
python scripts/train_model.py --force-register
```

---

**Report generated:** 2 September 2026
