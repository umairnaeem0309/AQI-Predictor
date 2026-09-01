"""
Live Data Fetcher — Fetches current weather and pollution from Open-Meteo.

For production predictions, the API needs current data for each city.
This module fetches real-time data and engineers features on-the-fly.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# City coordinates
CITIES = {
    "karachi": {"lat": 24.8607, "lon": 67.0011, "name": "Karachi"},
    "lahore": {"lat": 31.5204, "lon": 74.3587, "name": "Lahore"},
    "islamabad": {"lat": 33.6844, "lon": 73.0479, "name": "Islamabad"},
}

# Feature columns the model expects (71 features)
MODEL_FEATURES = None  # Loaded lazily from metadata


def _load_model_features():
    """Load the model's expected feature list."""
    global MODEL_FEATURES
    if MODEL_FEATURES is None:
        import json
        from pathlib import Path

        meta_path = Path("models/production/model_metadata.json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            MODEL_FEATURES = meta.get("feature_columns", [])
    return MODEL_FEATURES


def fetch_current_weather(city_id: str) -> Dict[str, Any]:
    """
    Fetch current weather from Open-Meteo.

    Args:
        city_id: City identifier (karachi, lahore, islamabad).

    Returns:
        Dictionary with weather values.
    """
    city = CITIES.get(city_id)
    if not city:
        raise ValueError(f"Unknown city: {city_id}")

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "current": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,cloud_cover,precipitation",
        "timezone": "UTC",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        current = data.get("current", {})
        return {
            "temperature": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "pressure": current.get("surface_pressure"),
            "wind_speed": current.get("wind_speed_10m"),
            "wind_direction": current.get("wind_direction_10m"),
            "cloud_cover": current.get("cloud_cover"),
            "precipitation": current.get("precipitation", 0.0),
        }
    except Exception as e:
        logger.error("Failed to fetch weather for %s: %s", city_id, e)
        raise


def fetch_current_pollution(city_id: str) -> Dict[str, Any]:
    """
    Fetch current air quality from Open-Meteo.

    Args:
        city_id: City identifier.

    Returns:
        Dictionary with pollution values.
    """
    city = CITIES.get(city_id)
    if not city:
        raise ValueError(f"Unknown city: {city_id}")

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "current": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "timezone": "UTC",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        current = data.get("current", {})
        return {
            "pm25": current.get("pm2_5"),
            "pm10": current.get("pm10"),
            "co": current.get("carbon_monoxide"),
            "no2": current.get("nitrogen_dioxide"),
            "so2": current.get("sulphur_dioxide"),
            "o3": current.get("ozone"),
        }
    except Exception as e:
        logger.error("Failed to fetch pollution for %s: %s", city_id, e)
        raise


def fetch_current_aqi(city_id: str) -> Dict[str, Any]:
    """
    Fetch current AQI from Open-Meteo (for reference only).

    Args:
        city_id: City identifier.

    Returns:
        Dictionary with AQI values.
    """
    city = CITIES.get(city_id)
    if not city:
        raise ValueError(f"Unknown city: {city_id}")

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "current": "us_aqi,us_aqi_pm2_5,us_aqi_pm10",
        "timezone": "UTC",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        current = data.get("current", {})
        return {
            "us_aqi": current.get("us_aqi"),
            "us_aqi_pm25": current.get("us_aqi_pm2_5"),
            "us_aqi_pm10": current.get("us_aqi_pm10"),
        }
    except Exception as e:
        logger.error("Failed to fetch AQI for %s: %s", city_id, e)
        raise


def fetch_historical_for_features(city_id: str, hours: int = 72) -> pd.DataFrame:
    """
    Fetch recent historical hourly data for feature engineering.

    Args:
        city_id: City identifier.
        hours: Number of hours to fetch (for lag/rolling features).

    Returns:
        DataFrame with hourly observations.
    """
    city = CITIES.get(city_id)
    if not city:
        raise ValueError(f"Unknown city: {city_id}")

    end_date = datetime.now(timezone.utc)
    start_date = pd.Timestamp(end_date) - pd.Timedelta(hours=hours + 24)

    # Fetch weather
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,cloud_cover,precipitation",
        "timezone": "UTC",
    }

    # Fetch pollution
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "timezone": "UTC",
    }

    try:
        weather_resp = requests.get(weather_url, params=weather_params, timeout=15)
        weather_resp.raise_for_status()
        weather_data = weather_resp.json()

        aq_resp = requests.get(aq_url, params=aq_params, timeout=15)
        aq_resp.raise_for_status()
        aq_data = aq_resp.json()

        # Build DataFrame
        weather_hourly = weather_data.get("hourly", {})
        aq_hourly = aq_data.get("hourly", {})

        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(weather_hourly.get("time", []), utc=True),
                "temperature": weather_hourly.get("temperature_2m", []),
                "humidity": weather_hourly.get("relative_humidity_2m", []),
                "pressure": weather_hourly.get("surface_pressure", []),
                "wind_speed": weather_hourly.get("wind_speed_10m", []),
                "wind_direction": weather_hourly.get("wind_direction_10m", []),
                "cloud_cover": weather_hourly.get("cloud_cover", []),
                "precipitation": weather_hourly.get("precipitation", []),
                "pm25": aq_hourly.get("pm2_5", []),
                "pm10": aq_hourly.get("pm10", []),
                "co": aq_hourly.get("carbon_monoxide", []),
                "no2": aq_hourly.get("nitrogen_dioxide", []),
                "so2": aq_hourly.get("sulphur_dioxide", []),
                "o3": aq_hourly.get("ozone", []),
            }
        )

        # Add location
        df["location_id"] = city_id
        df["city_name"] = city["name"]

        logger.info("Fetched %d hours of historical data for %s", len(df), city_id)
        return df

    except Exception as e:
        logger.error("Failed to fetch historical data for %s: %s", city_id, e)
        raise


def build_features_for_prediction(city_id: str) -> Dict[str, float]:
    """
    Build features for a live prediction.

    Fetches current + historical data, engineers features,
    and returns them in the format the model expects.

    Args:
        city_id: City identifier.

    Returns:
        Dictionary of feature values matching model's expected format.
    """
    from src.features.feature_engineering import (
        add_lag_features,
        add_rolling_features,
        add_time_features,
    )

    model_features = _load_model_features()
    if not model_features:
        raise ValueError("Model features not loaded")

    # Fetch historical data for lag/rolling features
    hist_df = fetch_historical_for_features(city_id, hours=96)

    # Add time features
    hist_df = add_time_features(hist_df)

    # Add lag features
    hist_df = add_lag_features(hist_df)

    # Add rolling features
    hist_df = add_rolling_features(hist_df)

    # Calculate AQI from PM2.5 and PM10
    from src.utils.epa_aqi import calculate_pm10_aqi, calculate_pm25_aqi

    hist_df["pm25_aqi"] = hist_df["pm25"].apply(
        lambda x: calculate_pm25_aqi(x) if pd.notna(x) else None
    )
    hist_df["pm10_aqi"] = hist_df["pm10"].apply(
        lambda x: calculate_pm10_aqi(x) if pd.notna(x) else None
    )
    hist_df["aqi"] = hist_df[["pm25_aqi", "pm10_aqi"]].max(axis=1)

    # Add AQI lag features
    for lag in [1, 6, 12, 24, 48, 72]:
        hist_df[f"aqi_lag_{lag}h"] = hist_df["aqi"].shift(lag)

    # Get the latest row with all features
    latest = hist_df.iloc[-1]

    # Build feature dictionary matching model's expected format
    features = {}
    for col in model_features:
        if col in latest.index:
            val = latest[col]
            features[col] = float(val) if pd.notna(val) else 0.0
        else:
            features[col] = 0.0  # Default for missing features

    logger.info("Built %d features for %s", len(features), city_id)
    return features


def get_live_prediction(city_id: str, model) -> Dict[str, Any]:
    """
    Get a live prediction for a city.

    Args:
        city_id: City identifier.
        model: Trained model.

    Returns:
        Dictionary with prediction results.
    """
    from src.models.confidence import predict_with_confidence
    from src.utils.aqi_categories import get_aqi_category

    # Build features
    features = build_features_for_prediction(city_id)

    # Make prediction
    X = np.array([list(features.values())])
    pred = model.predict(X)[0]

    # Parse results
    aqi_24h = int(pred[0])
    aqi_48h = int(pred[1])
    aqi_72h = int(pred[2])

    _, cat_24h = get_aqi_category(aqi_24h)
    _, cat_48h = get_aqi_category(aqi_48h)
    _, cat_72h = get_aqi_category(aqi_72h)

    # Compute confidence intervals
    try:
        ci = predict_with_confidence(pred.reshape(1, -1), confidence_level=90)
        intervals = ci.get("intervals", [])
        confidence = {
            "level": 90,
            "method": ci.get("interval_method", "residual_quantile"),
            "intervals": {
                "24h": intervals[0] if len(intervals) > 0 else None,
                "48h": intervals[1] if len(intervals) > 1 else None,
                "72h": intervals[2] if len(intervals) > 2 else None,
            },
        }
    except Exception as e:
        logger.warning(f"Failed to compute confidence intervals: {e}")
        confidence = None

    # Read model version from metadata
    model_version = "xgboost-v1.0"
    try:
        import json as _json
        from pathlib import Path as _Path

        meta_path = _Path("models/production/model_metadata.json")
        if meta_path.exists():
            with open(meta_path) as _f:
                meta = _json.load(_f)
            model_version = meta.get("model_version", model_version)
    except Exception:
        pass

    return {
        "city": CITIES[city_id]["name"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "aqi_24h": aqi_24h,
        "aqi_48h": aqi_48h,
        "aqi_72h": aqi_72h,
        "category_24h": cat_24h,
        "category_48h": cat_48h,
        "category_72h": cat_72h,
        "model_version": model_version,
        "confidence": confidence,
    }
