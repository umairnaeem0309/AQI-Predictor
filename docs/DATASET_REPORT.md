# Dataset Report

## AQI Predictor : Historical Dataset Analysis


## 1. Dataset Overview

| Metric | Value |
|--------|-------|
| Historical rows | 107,064 |
| Live hourly rows | accumulating since 2026-09-05 (+1/city/hour) |
| Cities | 3 (Karachi, Lahore, Islamabad) |
| Date range (historical) | 2022-08-03 to 2026-08-28 |
| Time span | **~4 years** |
| Rows per city (historical) | 35,688 |
| Weather features | 7 |
| Pollution features | 6 |
| Total features | 58 |
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
| Valid AQI values | 107,064 (100% of historical rows) |
| Range | 2 – 500 |
| Mean | 114.7 |
| Median | 100.0 |

### Per-City

| City | n | Mean | Median | Min | Max |
|------|---|------|--------|-----|-----|
| Islamabad | 35,688 | 105.6 | 97 | 3 | 343 |
| Karachi | 35,688 | 90.7 | 81 | 2 | 471 |
| Lahore | 35,688 | 147.8 | 148 | 3 | 500 |

---

## 4. Train/Val/Test Split

| Split | Rows | Percentage |
|-------|------|-----------|
| Train (72%) | 77,086 | 72% |
| Validation (8%) | 8,565 | 8% |
| Test (20%) | 21,413 | 20% |

**Split method:** Chronological (sorted by timestamp).

---

## 5. Data Quality

- **Duplicates:** 0
- **Missing values:** <0.2%
- **Negative pollutants:** 0 for PM2.5, PM10, CO, NO2, SO2 (91 negative O3 from CAMS model)

---

## 6. Feature Engineering

| Category | Count | Examples |
|----------|-------|----------|
| Weather | 7 | temperature, humidity, pressure, wind_speed, wind_direction, cloud_cover, precipitation |
| Pollution | 6 | pm25, pm10, co, no2, so2, o3 |
| AQI reference | 4 | aqi, us_aqi_open_meteo, us_aqi_pm25_open_meteo, us_aqi_pm10_open_meteo |
| Time | 7 | hour, day_of_week, month, is_weekend, season, hour_sin, hour_cos |
| Lag | 24 | aqi/pm25/temperature/humidity lags at 1h, 6h, 12h, 24h, 48h, 72h |
| Rolling | 10 | aqi mean(6h/12h/24h), std/min/max(24h), pm25 mean(6h/24h), temp/humidity mean(24h) |
| **Total** | **58** | (64 columns − 3 targets − timestamp − location_id − city_name) |

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

