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
