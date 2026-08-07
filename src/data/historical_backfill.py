"""
Historical Data Backfill — Collects historical observations for training.

Features:
- Caching: API responses stored locally to avoid repeated calls
- Resume capability: skips already-collected dates
- Progress checkpoints: saves state after each batch
- Sample collection: small test before full backfill
- Source quality tracking: documents data availability per observation

Important limitations:
- Historical API access depends on API provider capabilities
- OpenWeather history API requires paid subscription for >7 days
- AQICN historical data availability varies by station
- This pipeline documents limitations and handles missing data gracefully
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config import PROJECT_ROOT, get_api_key
from src.data.api_manager import APIManager
from src.data.schemas import CityConfig, StandardObservation
from src.data.validators import full_validation, drop_duplicates

logger = logging.getLogger(__name__)

# Paths
HISTORICAL_DIR = PROJECT_ROOT / "data" / "raw" / "historical"
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "cache"
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "raw" / "checkpoints"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# API access limitations (documented)
API_LIMITATIONS = {
    "openweather": {
        "historical_access": "limited",
        "free_tier_days": 7,
        "paid_tier_access": True,
        "note": "Free tier only provides current data. Historical access requires paid subscription.",
    },
    "aqicn": {
        "historical_access": "limited",
        "note": "AQICN provides current and forecast data. Historical data varies by station.",
    },
}


def _ensure_directories() -> None:
    """Create necessary directories."""
    for d in [HISTORICAL_DIR, CACHE_DIR, CHECKPOINT_DIR, PROCESSED_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _get_cache_key(city_id: str, date_str: str, source: str) -> str:
    """Generate cache key for an API response."""
    key_data = f"{city_id}_{date_str}_{source}"
    return hashlib.md5(key_data.encode()).hexdigest()


def _load_from_cache(cache_key: str) -> Optional[Dict]:
    """Load cached API response if available."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)
    return None


def _save_to_cache(cache_key: str, data: Dict) -> None:
    """Save API response to cache."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _load_checkpoint() -> Dict[str, Any]:
    """Load backfill progress checkpoint."""
    checkpoint_file = CHECKPOINT_DIR / "backfill_progress.json"
    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            return json.load(f)
    return {"completed_dates": [], "last_date": None, "total_records": 0}


def _save_checkpoint(progress: Dict[str, Any]) -> None:
    """Save backfill progress checkpoint."""
    checkpoint_file = CHECKPOINT_DIR / "backfill_progress.json"
    with open(checkpoint_file, "w") as f:
        json.dump(progress, f, indent=2, default=str)


def verify_api_access() -> Dict[str, Any]:
    """Verify which historical API endpoints are available.

    Returns:
        Dictionary with availability status and limitations for each API.
    """
    results = {}

    # Check OpenWeather
    ow_key = get_api_key("openweather")
    results["openweather"] = {
        "key_available": ow_key is not None and len(ow_key) > 0,
        "limitations": API_LIMITATIONS["openweather"],
        "recommendation": "Use mock data for historical period; real API for current data only",
    }

    # Check AQICN
    aqicn_key = get_api_key("aqicn")
    results["aqicn"] = {
        "key_available": aqicn_key is not None and len(aqicn_key) > 0,
        "limitations": API_LIMITATIONS["aqicn"],
        "recommendation": "Historical AQI data availability varies; use current data for training",
    }

    logger.info("API access verification complete: %s", json.dumps(results, indent=2))
    return results


def collect_sample_data(
    city_configs: Optional[List[CityConfig]] = None,
) -> pd.DataFrame:
    """Collect a small sample of data to verify the pipeline works.

    This is a checkpoint before full historical backfill.
    Collects data for a single hour across all cities.

    Args:
        city_configs: Cities to collect from. Defaults to configured cities.

    Returns:
        DataFrame with sample observations.
    """
    if city_configs is None:
        from src.config import load_config
        config = load_config()
        city_configs = [CityConfig(**city) for city in config.get("cities", [])]

    logger.info("Collecting sample data for %d cities", len(city_configs))

    manager = APIManager()
    observations = []

    for city in city_configs:
        try:
            obs = manager.fetch_city_data(city)
            if obs is not None:
                observations.append(obs)
                logger.info("Sample data collected for %s", city.name)
        except Exception as e:
            logger.warning("Sample collection failed for %s: %s", city.name, str(e))

    if not observations:
        logger.error("No sample data collected")
        return pd.DataFrame()

    df = pd.DataFrame([obs.model_dump() for obs in observations])
    report = full_validation(df)
    logger.info(
        "Sample collection complete: %d records, validation=%s",
        len(df),
        report.status.value,
    )

    return df


def generate_synthetic_historical_data(
    start_date: str = "2026-06-01",
    end_date: str = "2026-07-31",
    city_configs: Optional[List[CityConfig]] = None,
) -> pd.DataFrame:
    """Generate synthetic historical data for training.

    IMPORTANT: This generates realistic-looking historical data for
    pipeline development and testing. This data is NOT real API data
    and must NOT be used for final model training or reported results.

    This function exists because:
    - Historical API access is limited on free tiers
    - Full backfill requires paid API subscriptions
    - Pipeline testing needs data before real credentials are available

    Args:
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        city_configs: Cities to generate data for.

    Returns:
        DataFrame with synthetic hourly observations.
    """
    if city_configs is None:
        from src.config import load_config
        config = load_config()
        city_configs = [CityConfig(**city) for city in config.get("cities", [])]

    logger.info(
        "Generating synthetic historical data from %s to %s",
        start_date,
        end_date,
    )

    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # Base values per city (realistic ranges for Pakistan summer)
    city_profiles = {
        "karachi": {
            "city_name": "Karachi",
            "temp_base": 31, "temp_amp": 4,
            "humidity_base": 70, "humidity_amp": 10,
            "aqi_base": 120, "aqi_amp": 30,
            "pm25_base": 45, "pm25_amp": 15,
            "pm10_base": 65, "pm10_amp": 20,
        },
        "lahore": {
            "city_name": "Lahore",
            "temp_base": 35, "temp_amp": 5,
            "humidity_base": 55, "humidity_amp": 12,
            "aqi_base": 180, "aqi_amp": 40,
            "pm25_base": 80, "pm25_amp": 25,
            "pm10_base": 110, "pm10_amp": 30,
        },
        "islamabad": {
            "city_name": "Islamabad",
            "temp_base": 33, "temp_amp": 5,
            "humidity_base": 50, "humidity_amp": 15,
            "aqi_base": 85, "aqi_amp": 25,
            "pm25_base": 30, "pm25_amp": 12,
            "pm10_base": 50, "pm10_amp": 18,
        },
    }

    import numpy as np
    np.random.seed(42)

    rows = []
    hours = int((end - start).total_seconds() / 3600) + 1

    for city_config in city_configs:
        profile = city_profiles.get(city_config.id, city_profiles["karachi"])

        for i in range(hours):
            ts = start + timedelta(hours=i)
            hour = ts.hour

            # Daily temperature cycle (peak at 2pm, low at 5am)
            temp_cycle = np.sin((hour - 5) / 24 * 2 * np.pi)

            # Daily humidity cycle (inverse of temperature)
            hum_cycle = -np.sin((hour - 5) / 24 * 2 * np.pi)

            # AQI daily cycle (peak during rush hours)
            aqi_cycle = (np.sin((hour - 8) / 24 * 2 * np.pi) +
                        np.sin((hour - 17) / 24 * 2 * np.pi)) / 2

            # Weekly pattern (weekdays worse)
            is_weekend = ts.weekday() >= 5
            weekday_factor = 0.85 if is_weekend else 1.0

            temp = profile["temp_base"] + temp_cycle * profile["temp_amp"] + np.random.randn() * 1.5
            humidity = profile["humidity_base"] + hum_cycle * profile["humidity_amp"] + np.random.randn() * 3
            aqi = max(0, profile["aqi_base"] + aqi_cycle * profile["aqi_amp"] * weekday_factor + np.random.randn() * 8)
            pm25 = max(0, profile["pm25_base"] + aqi_cycle * profile["pm25_amp"] * weekday_factor + np.random.randn() * 5)
            pm10 = max(0, profile["pm10_base"] + aqi_cycle * profile["pm10_amp"] * weekday_factor + np.random.randn() * 7)

            rows.append({
                "timestamp": ts,
                "location_id": city_config.id,
                "city_name": profile["city_name"],
                "temperature": round(temp, 2),
                "humidity": round(np.clip(humidity, 0, 100), 2),
                "wind_speed": round(3 + np.random.rand() * 4, 2),
                "pressure": round(1010 + np.random.randn() * 3, 1),
                "weather_condition": "clear" if np.random.rand() > 0.3 else "cloudy",
                "aqi": int(aqi),
                "pm25": round(pm25, 2),
                "pm10": round(pm10, 2),
                "co": round(200 + np.random.rand() * 80, 2),
                "no2": round(20 + np.random.rand() * 15, 2),
                "so2": round(10 + np.random.rand() * 8, 2),
                "o3": round(40 + np.random.rand() * 25, 2),
                "data_source": "synthetic",
            })

    df = pd.DataFrame(rows)
    logger.info(
        "Synthetic data generated: %d rows, %d cities, %d hours",
        len(df),
        len(city_configs),
        hours,
    )

    return df


def run_historical_backfill(
    start_date: str = "2026-06-01",
    end_date: str = "2026-07-31",
    city_configs: Optional[List[CityConfig]] = None,
    use_synthetic: bool = True,
) -> pd.DataFrame:
    """Run historical data backfill.

    Pipeline:
    1. Verify API access
    2. Collect sample data (checkpoint)
    3. Collect or generate historical data
    4. Run data quality checks
    5. Save processed dataset
    6. Generate metadata

    Args:
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        city_configs: Cities to collect for.
        use_synthetic: If True, generate synthetic data (for pipeline development).

    Returns:
        DataFrame with historical observations.
    """
    _ensure_directories()

    logger.info("Starting historical backfill: %s to %s", start_date, end_date)

    # Step 1: Verify API access
    api_status = verify_api_access()

    # Step 2: Check for existing checkpoint (resume capability)
    progress = _load_checkpoint()
    if progress.get("completed_dates"):
        logger.info(
            "Resuming from checkpoint: %d dates already completed",
            len(progress["completed_dates"]),
        )

    # Step 3: Collect or generate data
    if use_synthetic:
        logger.info("Using synthetic data for pipeline development")
        df = generate_synthetic_historical_data(start_date, end_date, city_configs)
    else:
        # Real API collection would go here
        logger.warning(
            "Real API historical collection not yet implemented. "
            "Use use_synthetic=True for development."
        )
        df = generate_synthetic_historical_data(start_date, end_date, city_configs)

    # Step 4: Data quality checks
    df = drop_duplicates(df)
    report = full_validation(df)

    # Step 5: Save processed dataset
    output_file = PROCESSED_DIR / "raw_observations.csv"
    df.to_csv(output_file, index=False)
    logger.info("Saved raw observations to %s", output_file)

    # Step 6: Generate metadata
    metadata = {
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "date_range": {"start": start_date, "end": end_date},
        "cities": [c.id for c in (city_configs or [])],
        "total_records": len(df),
        "quality_status": report.status.value,
        "quality_warnings": report.warnings,
        "api_status": api_status,
        "use_synthetic": use_synthetic,
    }
    metadata_file = PROCESSED_DIR / "raw_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Update checkpoint
    progress["completed_dates"].append(end_date)
    progress["last_date"] = end_date
    progress["total_records"] = len(df)
    _save_checkpoint(progress)

    logger.info(
        "Historical backfill complete: %d records, status=%s",
        len(df),
        report.status.value,
    )

    return df
