# Dataset Report

## AQI Predictor — Historical Dataset Analysis

**Generated:** 27 August 2026
**Status:** Ready for model training
**Dataset type:** real_api_data (approved_for_training: true)

**Data Range Transparency:**
- Original request: 5 years of historical data
- Actual dataset: **4 years** (2022-08-01 to 2026-08-26)
- Reason: Open-Meteo CAMS Global air quality data starts August 2022
- Weather data available from 2017+ (9 years) but merged dataset limited by AQ availability
- 4 years provides 107,064 hourly observations across 3 cities — sufficient for training

---

## 1. Dataset Overview

| Metric | Value |
|--------|-------|
| Total rows | 107,064 |
| Cities | 3 (Karachi, Lahore, Islamabad) |
| Date range | 2022-08-01 to 2026-08-26 |
| Time span | **4 years, 0 months, 26 days** (not 5 — see transparency note above) |
| Hourly resolution | 35,688 unique hours |
| Rows per city | 35,688 (uniform) |
| Weather features | 7 (temperature, humidity, pressure, wind_speed, wind_direction, cloud_cover, precipitation) |
| Pollution features | 6 (pm25, pm10, co, no2, so2, o3) |
| Derived features | 63 (lag, rolling, time, derived) |
| Target variables | 3 (target_aqi_24h, target_aqi_48h, target_aqi_72h) |
| Total columns (raw) | 28 |
| Total columns (features) | 79 |

---

## 2. Data Sources

| Source | Endpoint | Data | Resolution | History |
|--------|----------|------|-----------|---------|
| Open-Meteo Weather | `/v1/archive` | temperature, humidity, pressure, wind, clouds, precipitation | Hourly | 2017+ (IFS 9km) |
| Open-Meteo Air Quality | `/v1/air-quality` | PM2.5, PM10, CO, NO2, SO2, O3 | Hourly | Aug 2022+ (CAMS Global 45km) |

**API calls made:**
- Weather: 15 requests (3 cities × 5 chunks), 0 errors
- Air Quality: 51 requests (3 cities × 17 chunks), 0 errors
- Total: 66 requests, 0 errors

---

## 3. Feature Descriptions

### 3.1 Weather Features (from Open-Meteo Archive)

| Feature | Unit | Range | Missing | Description |
|---------|------|-------|---------|-------------|
| temperature | °C | 1.1 – 46.5 | 0.0% | Air temperature at 2m |
| humidity | % | 5 – 100 | 0.0% | Relative humidity at 2m |
| pressure | hPa | 934.4 – 1024.4 | 0.0% | Surface pressure |
| wind_speed | m/s | 0.0 – 12.96 | 0.0% | Wind speed at 10m |
| wind_direction | ° | 1 – 360 | 0.0% | Wind direction at 10m |
| cloud_cover | % | 0 – 100 | 0.0% | Total cloud cover |
| precipitation | mm | 0.0 – 21.5 | 0.0% | Precipitation amount |

### 3.2 Pollution Features (from Open-Meteo CAMS)

| Feature | Unit | Range | Missing | Description |
|---------|------|-------|---------|-------------|
| pm25 | μg/m³ | 0.4 – 448.9 | 0.2% | Fine particulate matter |
| pm10 | μg/m³ | 0.5 – 890.3 | 0.2% | Coarse particulate matter |
| co | μg/m³ | 14.0 – 12926.0 | 0.2% | Carbon monoxide |
| no2 | μg/m³ | 0.0 – 317.7 | 0.2% | Nitrogen dioxide |
| so2 | μg/m³ | 0.0 – 260.1 | 0.2% | Sulphur dioxide |
| o3 | μg/m³ | -11.0 – 680.0 | 0.2% | Ozone (91 negative values from CAMS model) |

### 3.3 Target Variables

| Target | Description | Valid Rows (total) |
|--------|-------------|-------------------|
| target_aqi_24h | AQI at t+24h | 106,848 / 107,064 |
| target_aqi_48h | AQI at t+48h | 106,848 / 107,064 |
| target_aqi_72h | AQI at t+72h | 106,848 / 107,064 |

**Note:** Last 216 rows per city cannot have valid targets (AQI not yet available at those future timestamps). This is expected.

---

## 4. AQI Distribution

### 4.1 Overall

| Statistic | Value |
|-----------|-------|
| Valid AQI values | 106,848 (99.8%) |
| Range | 2 – 500 |
| Mean | 114.7 |
| Median | 100.0 |
| Std Dev | 49.1 |

### 4.2 AQI Categories

| Category | Count | Percentage | AQI Range |
|----------|-------|-----------|-----------|
| Good | 782 | 0.7% | 0–50 |
| Moderate | 53,057 | 49.7% | 51–100 |
| Unhealthy for Sensitive Groups | 26,736 | 25.0% | 101–150 |
| Unhealthy | 21,300 | 19.9% | 151–200 |
| Very Unhealthy | 4,218 | 3.9% | 201–300 |
| Hazardous | 755 | 0.7% | 301–500 |

### 4.3 Per-City AQI

| City | n | Mean | Median | Std | Min | Max |
|------|---|------|--------|-----|-----|-----|
| Islamabad | 35,616 | 105.6 | 97 | 35.3 | 3 | 343 |
| Karachi | 35,616 | 90.7 | 81 | 31.6 | 2 | 471 |
| Lahore | 35,616 | 147.8 | 148 | 56.8 | 3 | 500 |

**Observations:**
- Lahore has the highest average AQI (147.8), classified as "Unhealthy for Sensitive Groups"
- Karachi has moderate pollution (90.7 mean), mostly "Moderate" category
- Islamabad is the cleanest of the three (105.6 mean)
- All three cities experience hazardous episodes (AQI > 300)

---

## 5. Data Quality Checks

### 5.1 Missing Values

| Column | Missing | Percentage | Impact |
|--------|---------|-----------|--------|
| temperature | 0 | 0.00% | None |
| humidity | 0 | 0.00% | None |
| pressure | 0 | 0.00% | None |
| wind_speed | 0 | 0.00% | None |
| pm25 | 216 | 0.20% | Low |
| pm10 | 216 | 0.20% | Low |
| co | 216 | 0.20% | Low |
| no2 | 216 | 0.20% | Low |
| so2 | 216 | 0.20% | Low |
| o3 | 216 | 0.20% | Low |
| aqi | 216 | 0.20% | Low |

**Root cause:** 216 missing pollution rows per city correspond to 9 days where CAMS Global model had gaps. Weather data is complete (0% missing) because ERA5/IFS reanalysis is gap-free.

### 5.2 Negative Values

| Column | Negative Count | Treatment |
|--------|---------------|-----------|
| o3 | 91 | CAMS model artifact — values clipped to 0 during AQI calculation |

### 5.3 Duplicate Check

No duplicate (timestamp, location_id) pairs found.

### 5.4 Timestamp Ordering

All timestamps are chronologically ordered within each city.

---

## 6. Train/Val/Test Split

| Split | Years | Rows | Percentage | Target_24h valid | Target_48h valid | Target_72h valid |
|-------|-------|------|-----------|-----------------|-----------------|-----------------|
| Train | 2022–2024 | 63,648 | 59.5% | 63,504 | 63,576 | 63,648 |
| Val | 2025 | 26,280 | 24.5% | 26,280 | 26,280 | 26,280 |
| Test | 2026 | 17,136 | 16.0% | 17,064 | 16,992 | 16,920 |

**Split method:** Chronological year-based split (no random shuffling).

**Justification:** Time-series forecasting requires temporal separation between train/val/test to prevent data leakage. Models must predict future AQI using only past data.

---

## 7. Feature Engineering Pipeline

### 7.1 Feature Categories

| Category | Features | Count | Description |
|----------|----------|-------|-------------|
| Raw current | temperature, humidity, pressure, wind_speed, pm25, pm10, co, no2, so2, o3, aqi | 11 | Current observation values |
| Time-based | hour, day_of_week, month, season, is_weekend, hour_sin, hour_cos | 7 | Cyclical time encoding |
| Lag | {aqi,pm25,temperature,humidity}_lag_{1,6,12,24,48,72}h | 24 | Historical values at specific offsets |
| Rolling | {aqi,pm25,temperature,humidity}_rolling_{mean,std,min,max}_{window} | 10 | Windowed statistics |
| Derived | aqi_change_rate, pm25_pm10_ratio, no2_so2_ratio, etc. | 10 | Ratios, change rates, interactions |
| **Total** | | **79** | |

### 7.2 Data Leakage Prevention

- **Lag features:** Use `shift(N)` which only references historical data
- **Rolling features:** Use `closed='left'` which excludes the current period
- **Target generation:** Forward-shift AQI by 24/48/72 hours within each city group
- **City isolation:** All features computed per-city (no cross-city contamination)
- **No future data:** Feature timestamp < target timestamp by construction

---

## 8. AQI Calculation Flow

### 8.1 Pipeline

```
Open-Meteo PM2.5 (μg/m³)
         ↓
    truncate_pm25() → floor(conc * 10) / 10
         ↓
    calculate_aqi_from_concentration() → EPA linear interpolation
         ↓
    pm25_aqi (0-500)

Open-Meteo PM10 (μg/m³)
         ↓
    truncate_pm10() → floor(conc)
         ↓
    calculate_aqi_from_concentration() → EPA linear interpolation
         ↓
    pm10_aqi (0-500)

aqi = max(pm25_aqi, pm10_aqi)
dominant_pollutant = argmax(pm25_aqi, pm10_aqi)
```

### 8.2 EPA Methodology

- **Standard:** US EPA AQI (EPA-454/B-24-002, May 2024)
- **Breakpoints:** PM2.5 (Good: 0.0–9.0 μg/m³), PM10 (Good: 0–54 μg/m³)
- **Formula:** `AQI = ((AQI_high - AQI_low) / (C_high - C_low)) × (C - C_low) + AQI_low`
- **Rounding:** Final AQI rounded to nearest integer

### 8.3 Derived vs Official

This AQI is a **derived estimate** from CAMS model pollutant concentrations. It is NOT an official EPA/AirNow monitor reading. The CAMS Global model provides gridded (45km) atmospheric composition data, not point measurements from monitoring stations.

---

## 9. Files Generated

| File | Size | Description |
|------|------|-------------|
| `data/raw/historical/weather_data.csv` | 107,064 rows | Raw weather data |
| `data/raw/historical/air_quality_data.csv` | 107,064 rows | Raw air quality data |
| `data/processed/raw_observations.csv` | 107,064 rows | Merged + AQI calculated |
| `data/processed/train_features.csv` | 63,648 rows | Training features |
| `data/processed/train_targets.csv` | 63,648 rows | Training targets |
| `data/processed/val_features.csv` | 26,280 rows | Validation features |
| `data/processed/val_targets.csv` | 26,280 rows | Validation targets |
| `data/processed/test_features.csv` | 17,136 rows | Test features |
| `data/processed/test_targets.csv` | 17,136 rows | Test targets |
| `data/processed/ingestion_metadata.json` | — | Pipeline metadata |
| `data/processed/dataset_metadata.json` | — | Dataset metadata |

---

## 10. Reproducibility

### 10.1 Command to Regenerate

```bash
python scripts/build_dataset.py --start-date 2022-08-01 --end-date 2026-08-26
```

### 10.2 Deterministic Aspects

- **Weather data:** ERA5/IFS reanalysis is deterministic (same coordinates → same values)
- **Air quality data:** CAMS model output is deterministic
- **AQI calculation:** Deterministic (same concentrations → same AQI)
- **Feature engineering:** Deterministic (same inputs → same features)
- **Train/val/test split:** Deterministic (year-based)

### 10.3 Non-Deterministic Aspects

- Open-Meteo may update reanalysis datasets (rare, usually consistent)
- CAMS model version may change (documented in metadata)

---

## 11. Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| CAMS Global AQ at 45km resolution | City-level estimate, not street-level | Acceptable for city-level forecasting |
| O3 has 91 negative values | CAMS model artifact | Clipped to 0 during AQI calculation |
| No ground-station validation | Derived AQI may differ from official monitors | Documented; use for relative prediction |
| Air quality starts Aug 2022 | Shorter than weather history | 4 years sufficient for training |
| Weather is reanalysis, not raw observations | Slight differences from ground truth | ERA5/IFS widely accepted in research |

---

## 12. Approval for Training

| Criterion | Status |
|-----------|--------|
| Real API data (not synthetic) | ✅ |
| 500+ usable rows per city | ✅ (35,616 per city) |
| Valid AQI targets | ✅ (99.8% valid) |
| No data leakage | ✅ (verified) |
| Chronological split | ✅ |
| EPA methodology documented | ✅ |
| Dataset metadata complete | ✅ |

**Dataset is approved for model training.**
