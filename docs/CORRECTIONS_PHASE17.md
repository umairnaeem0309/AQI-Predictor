# Phase 17 Corrections — 26 August 2026

## Corrections Applied

### 1. Python Environment Fixed
- Created clean Python 3.11.15 environment via conda (`aqi-predictor`)
- All dependencies installed within pinned ranges
- duckdb 1.0.0, hopsworks 5.0.6, mlflow 2.22.5 verified

### 2. AQICN Staleness Root Cause Identified and Fixed
- **Root cause**: City-level feeds (`/feed/karachi/`) return stale cached data
- **Fix**: Use bound station IDs (`/feed/@7393/`) which return fresh data
- **Bound stations**: Karachi @7393, Lahore @7432, Islamabad @7433
- **Freshness**: Bound station data is 6-7 hours old (vs months for city feeds)

### 3. Source-Level Freshness Validation Added
- `raw_response_time` field stores provider observation timestamp
- `collected_at` field stores local collection timestamp
- `is_training_valid` flag marks stale observations
- Quality gate now uses `raw_response_time` for freshness checks
- Data sufficiency only counts training-valid observations

### 4. Original Three Observations Treatment
- 9 total observations in master CSV (3 original stale + 3 stale + 3 fresh)
- Only 3 observations are training-valid (fresh OpenWeather data)
- 6 observations marked as NOT training-valid (stale AQICN data)
- Original observations preserved in audit trail but excluded from training

### 5. Collection Cadence Analysis
- Hourly collection: 6 OpenWeather + 3 AQICN = 9 calls/hour
- Daily: 144 OpenWeather + 72 AQICN = 216 calls/day
- Both within free tier limits (1,000 calls/day each)
- 21 days × 24 hours × 3 cities = 1,512 potential observations

### 6. AQICN Update Frequency Limitation
- AQICN ground stations update every 6-8 hours
- For hourly training data, weather features come from OpenWeather (always fresh)
- AQI targets from AQICN will lag by 6-8 hours
- This is an inherent limitation of the free-tier AQICN service

### 7. Historical Data Wording Corrected
- "Historical backfill is not available through the currently tested
  endpoints/credentials" (not a universal claim)

### 8. Collection Start Date
- Official real collection start: NOT YET SET
- Requires fresh AQI observations from bound stations over 21+ days
- Earliest possible training readiness: ~17-26 September 2026

## Files Modified
- `src/data/aqicn_client.py` — bound station IDs, freshness validation
- `src/data/schemas.py` — collected_at, is_training_valid, OPENWEATHER_AQICN
- `src/data/openweather_client.py` — collected_at field
- `scripts/quality_gate.py` — source timestamp freshness, training validity
- `src/models/registry.py` — missing imports
- `requirements.txt` — hopsworks 5.0.x, scikit-learn, xgboost ranges
- `tests/unit/test_aqicn_client.py` — bound station mock URLs
- `tests/unit/test_openweather_client.py` — test fixes

## Commits Created
1. `b23cb93` — fix: AQICN bound stations + freshness validation
2. `b95023c` — fix: quality gate source timestamp freshness
3. `68fc022` — test: fix tests for bound stations + missing imports
4. `a6b51e6` — chore: update hopsworks + validation scripts

## Remaining Blockers for Phase 18
1. Collect 21+ days of fresh real data (hourly collection)
2. Verify Hopsworks feature store integration (connection works, feature groups empty)
3. Resolve Python environment (3.11 active, but project needs proper venv setup)
4. 26 pre-existing test failures need investigation (not Phase 17 related)
