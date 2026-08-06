# Data Dictionary

## AQI Predictor — Field Definitions and Data Contracts

**Version:** 1.0  
**Date:** 2 August 2026  
**Status:** Phase 2 — Data Collection Architecture  

---

## 1. Overview

This document defines all data fields used in the AQI Predictor system, including:
- Raw API response fields (from OpenWeather and AQICN)
- Normalized standard observation schema
- Source ownership (which API is authoritative for each field)
- Missing value handling strategy

---

## 2. Source Ownership Rules

| Field Category | Authoritative Source | Fallback Source |
|---|---|---|
| Weather (temperature, humidity, wind, pressure, condition) | OpenWeather | None |
| AQI (US EPA scale) | AQICN | OpenWeather (1-5 scale, different) |
| Pollutants (PM2.5, PM10, CO, NO2, SO2, O3) | AQICN | OpenWeather |
| Timestamps | Both (UTC normalized) | — |

**Rule:** When both sources provide a field, AQICN values take precedence for AQI/pollution. OpenWeather values take precedence for weather fields.

---

## 3. Standard Observation Schema

This is the canonical data contract between data collection and feature engineering layers.

### 3.1 Metadata Fields

| Field | Type | Required | Description | Source |
|---|---|---|---|---|
| `timestamp` | datetime (UTC) | Yes | Observation time in UTC | Both |
| `location_id` | str | Yes | City identifier (karachi, lahore, islamabad) | Config |
| `city_name` | str | Yes | Human-readable city name | API |
| `data_source` | str | Yes | Source API (openweather, aqicn) | Client |
| `raw_response_time` | datetime (UTC) | No | Original API timestamp before normalization | API |

### 3.2 Weather Fields

| Field | Type | Unit | Required | Description | Source |
|---|---|---|---|---|---|
| `temperature` | float | °C | No | Current temperature | OpenWeather (authoritative) |
| `humidity` | float | % | No | Relative humidity | OpenWeather (authoritative) |
| `wind_speed` | float | m/s | No | Wind speed | OpenWeather (authoritative) |
| `pressure` | float | hPa | No | Atmospheric pressure | OpenWeather (authoritative) |
| `weather_condition` | str | — | No | Weather description (e.g., "few clouds") | OpenWeather (authoritative) |

### 3.3 AQI/Pollution Fields

| Field | Type | Unit | Required | Description | Source |
|---|---|---|---|---|---|
| `aqi` | int | US EPA 0-500 | No | Air Quality Index | AQICN (authoritative) |
| `pm25` | float | μg/m³ | No | Fine particulate matter | AQICN (authoritative) |
| `pm10` | float | μg/m³ | No | Coarse particulate matter | AQICN (authoritative) |
| `co` | float | μg/m³ | No | Carbon monoxide | AQICN (authoritative) |
| `no2` | float | μg/m³ | No | Nitrogen dioxide | AQICN (authoritative) |
| `so2` | float | μg/m³ | No | Sulfur dioxide | AQICN (authoritative) |
| `o3` | float | μg/m³ | No | Ozone | AQICN (authoritative) |

---

## 4. Raw API Response Fields

### 4.1 OpenWeather /data/2.5/weather

| JSON Path | Type | Maps To | Description |
|---|---|---|---|
| `main.temp` | float | `temperature` | Temperature in Celsius (metric) |
| `main.humidity` | int | `humidity` | Relative humidity (%) |
| `main.pressure` | int | `pressure` | Atmospheric pressure (hPa) |
| `wind.speed` | float | `wind_speed` | Wind speed (m/s) |
| `weather[0].description` | str | `weather_condition` | Weather description |
| `name` | str | `city_name` | City name |
| `dt` | int | `timestamp` | Unix timestamp (UTC) |
| `timezone` | int | — | Timezone offset from UTC (seconds) |
| `coord.lat` | float | — | Latitude |
| `coord.lon` | float | — | Longitude |

### 4.2 OpenWeather /data/2.2/air_pollution

| JSON Path | Type | Maps To | Description |
|---|---|---|---|
| `list[0].main.aqi` | int | — | OpenWeather AQI (1-5 scale, NOT US EPA) |
| `list[0].components.pm2_5` | float | `pm25` | PM2.5 (μg/m³) |
| `list[0].components.pm10` | float | `pm10` | PM10 (μg/m³) |
| `list[0].components.co` | float | `co` | CO (μg/m³) |
| `list[0].components.no2` | float | `no2` | NO2 (μg/m³) |
| `list[0].components.so2` | float | `so2` | SO2 (μg/m³) |
| `list[0].components.o3` | float | `o3` | O3 (μg/m³) |
| `list[0].dt` | int | — | Measurement time (Unix timestamp) |

**Note:** OpenWeather AQI uses a 1-5 scale. This is different from US EPA AQI (0-500). OpenWeather AQI is stored as `aqi_ow` internally and is not directly used as the primary AQI value.

### 4.3 AQICN/WAQI /feed/{city_id}/

| JSON Path | Type | Maps To | Description |
|---|---|---|---|
| `data.aqi` | int | `aqi` | US EPA AQI (0-500) |
| `data.iaqi.pm25.v` | float | `pm25` | PM2.5 (μg/m³) |
| `data.iaqi.pm10.v` | float | `pm10` | PM10 (μg/m³) |
| `data.iaqi.co.v` | float | `co` | CO (μg/m³) |
| `data.iaqi.no2.v` | float | `no2` | NO2 (μg/m³) |
| `data.iaqi.so2.v` | float | `so2` | SO2 (μg/m³) |
| `data.iaqi.o3.v` | float | `o3` | O3 (μg/m³) |
| `data.time.v` | int | `timestamp` | Unix timestamp |
| `data.time.iso` | str | `timestamp` | ISO 8601 timestamp |
| `data.city.name` | str | `city_name` | City name |

---

## 5. Missing Value Handling

### 5.1 Strategy

| Scenario | Handling |
|---|---|
| Weather field missing from OpenWeather | Set to None; feature engineering handles missing values |
| Pollutant missing from AQICN | Fall back to OpenWeather value; if also missing, set to None |
| AQI missing from both sources | Set to None; feature engineering skips this observation |
| Timestamp missing | Use current UTC time; log warning |
| Entire API call fails | Log error; continue with available data |
| AQICN weather fields (always missing) | Set to None; OpenWeather is authoritative |

### 5.2 Feature Engineering Impact

Feature engineering must handle None values:
- Rolling averages skip None values
- Lag features use last available value
- Missing pollutant ratios are set to None
- Model training drops observations with too many missing values

---

## 6. Timezone Normalization

All timestamps are normalized to UTC:
- OpenWeather `dt` field is UTC (timezone offset is informational only)
- AQICN `time.v` is UTC; `time.iso` may include timezone offset
- All datetime objects use `datetime.timezone.utc`
- Feature engineering works exclusively in UTC

---

## 7. Data Freshness Requirements

| Data Type | Maximum Age | Action if Stale |
|---|---|---|
| Live weather | 2 hours | Log warning; still use data |
| Live AQI | 2 hours | Log warning; still use data |
| AQICN data | 2 hours | Log staleness warning |
| Historical backfill | N/A | No freshness check |

---

## 8. Mock Data (API-Shaped Responses Only)

Mock data files in `data/mock/` contain API-shaped JSON responses for testing.

**Purpose:** Unit tests, CI/CD, pipeline validation without API access.

**Boundary:** Mock data must NEVER be used for:
- Final model training
- Reported evaluation metrics
- Production data

| File | Description |
|---|---|
| `openweather_response_karachi.json` | Sample weather response for Karachi |
| `openweather_response_lahore.json` | Sample weather response for Lahore |
| `openweather_response_islamabad.json` | Sample weather response for Islamabad |
| `openweather_pollution_karachi.json` | Sample pollution response for Karachi |
| `aqicn_response_karachi.json` | Sample AQICN response for Karachi |
| `aqicn_response_lahore.json` | Sample AQICN response for Lahore |
| `aqicn_response_islamabad.json` | Sample AQICN response for Islamabad |

---

## 9. Engineered Features

**Feature Version:** 1.0.0  
**Schema Version:** 1.0  
**Pipeline:** `src/features/feature_engineering.py`  
**Total Features:** 37

### 9.1 Time Features (7 features)

| Feature | Type | Calculation | Availability | Purpose |
|---|---|---|---|---|
| `hour` | int | `timestamp.hour` | t (immediate) | Daily pollution cycles |
| `day_of_week` | int | `timestamp.weekday()` (0=Mon, 6=Sun) | t (immediate) | Weekly patterns |
| `month` | int | `timestamp.month` | t (immediate) | Seasonal patterns |
| `season` | int | 0=Winter, 1=Spring, 2=Summer, 3=Fall | t (immediate) | Broad seasonal classification |
| `is_weekend` | int | 1 if day_of_week >= 5, else 0 | t (immediate) | Weekend activity patterns |
| `hour_sin` | float | `sin(2π × hour / 24)` | t (immediate) | Cyclical encoding |
| `hour_cos` | float | `cos(2π × hour / 24)` | t (immediate) | Cyclical encoding |

### 9.2 Lag Features (12 features)

| Feature | Source | Lag | Availability | Purpose |
|---|---|---|---|---|
| `aqi_lag_1h` | AQI | 1 hour | t (historical) | Short-term AQI momentum |
| `aqi_lag_6h` | AQI | 6 hours | t (historical) | Medium-term AQI trend |
| `aqi_lag_12h` | AQI | 12 hours | t (historical) | Half-day pattern |
| `aqi_lag_24h` | AQI | 24 hours | t (historical) | Daily pattern (same time yesterday) |
| `aqi_lag_48h` | AQI | 48 hours | t (historical) | Two-day pattern |
| `aqi_lag_72h` | AQI | 72 hours | t (historical) | Three-day pattern |
| `pm25_lag_1h` | PM2.5 | 1 hour | t (historical) | Short-term PM2.5 trend |
| `pm25_lag_24h` | PM2.5 | 24 hours | t (historical) | Daily PM2.5 pattern |
| `temperature_lag_1h` | Temperature | 1 hour | t (historical) | Temperature momentum |
| `temperature_lag_24h` | Temperature | 24 hours | t (historical) | Daily temperature cycle |
| `humidity_lag_1h` | Humidity | 1 hour | t (historical) | Humidity momentum |
| `humidity_lag_24h` | Humidity | 24 hours | t (historical) | Daily humidity cycle |

**Note:** All lag features use data from t-N hours ago. They are available at prediction time t.

### 9.3 Rolling Window Features (10 features)

| Feature | Source | Window | Aggregation | Availability | Purpose |
|---|---|---|---|---|---|
| `aqi_rolling_mean_6h` | AQI | 6 hours | Mean | t (historical window) | Short-term average |
| `aqi_rolling_mean_12h` | AQI | 12 hours | Mean | t (historical window) | Half-day average |
| `aqi_rolling_mean_24h` | AQI | 24 hours | Mean | t (historical window) | Daily average |
| `aqi_rolling_std_24h` | AQI | 24 hours | Std Dev | t (historical window) | AQI volatility |
| `aqi_rolling_min_24h` | AQI | 24 hours | Min | t (historical window) | Best AQI in period |
| `aqi_rolling_max_24h` | AQI | 24 hours | Max | t (historical window) | Worst AQI in period |
| `pm25_rolling_mean_6h` | PM2.5 | 6 hours | Mean | t (historical window) | Short-term PM2.5 average |
| `pm25_rolling_mean_24h` | PM2.5 | 24 hours | Mean | t (historical window) | Daily PM2.5 average |
| `temperature_rolling_mean_24h` | Temperature | 24 hours | Mean | t (historical window) | Daily temperature average |
| `humidity_rolling_mean_24h` | Humidity | 24 hours | Mean | t (historical window) | Daily humidity average |

**Note:** Rolling windows use `closed='left'` to exclude the current period (no future data leakage).

### 9.4 Derived Features (10 features)

| Feature | Calculation | Availability | Purpose |
|---|---|---|---|
| `aqi_change_rate_1h` | `aqi - aqi_lag_1h` | t (historical) | Hourly AQI change speed |
| `aqi_change_rate_6h` | `(aqi - aqi_lag_6h) / 6` | t (historical) | 6-hour AQI change speed |
| `aqi_change_rate_24h` | `(aqi - aqi_lag_24h) / 24` | t (historical) | Daily AQI change speed |
| `aqi_trend_24h` | `aqi_rolling_mean_6h - aqi_rolling_mean_24h` | t (historical) | Short vs long term direction |
| `pm25_pm10_ratio` | `pm25 / pm10` | t (immediate) | Particle size distribution |
| `no2_so2_ratio` | `no2 / so2` | t (immediate) | Industrial vs traffic signature |
| `o3_no2_ratio` | `o3 / no2` | t (immediate) | Photochemical activity |
| `temp_humidity_interaction` | `temperature × humidity / 100` | t (immediate) | Heat index approximation |
| `wind_cooling_effect` | `temperature - (wind_speed × 2)` | t (immediate) | Wind-chill approximation |
| `aqi_deviation_from_24h_avg` | `aqi - aqi_rolling_mean_24h` | t (historical) | Deviation from recent average |

**Note:** Feature usefulness will be experimentally evaluated during model training.

### 9.5 Feature Metadata

Every generated feature dataset includes:
- `feature_version`: Semantic version of the feature definitions
- `schema_version`: Schema version for compatibility
- `generation_timestamp`: UTC timestamp when features were generated
- `source_row_count`: Number of input observations
- `feature_count`: Number of features generated |
