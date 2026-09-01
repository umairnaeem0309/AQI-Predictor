"""
Historical Data Route

Serves historical AQI, weather, and pollutant data for analytics.
Reads from local CSV files (no external calls).
"""

import logging
import os
from datetime import datetime
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from app.backend.dependencies import verify_api_key
from app.schemas.responses import ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data", tags=["data"])

# Path to processed data
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")


def _load_dataset():
    """Load dataset from Hopsworks (primary) or local CSV (fallback).

    Returns DataFrame or None if no data available.
    """
    import os as _os

    # Try Hopsworks first — use direct connection
    try:
        import hopsworks

        host = _os.environ.get("HOPSWORKS_HOST")
        api_key = _os.environ.get("HOPSWORKS_API_KEY")
        project_name = _os.environ.get("HOPSWORKS_PROJECT", "AQI_Predictor")

        if host and api_key:
            project = hopsworks.login(
                host=host,
                api_key_value=api_key,
                project=project_name,
            )
            fs = project.get_feature_store()
            for fg_name in ["aqi_features_prod", "aqi_features_test"]:
                try:
                    fg = fs.get_feature_group(name=fg_name, version=1)
                    df = fg.read()
                    if df is not None and not df.empty:
                        if "timestamp" in df.columns:
                            df["timestamp"] = pd.to_datetime(
                                df["timestamp"], utc=True, errors="coerce"
                            )
                        logger.info(f"Loaded {len(df)} rows from Hopsworks")
                        return df
                except Exception:
                    continue
    except Exception as e:
        logger.warning(f"Hopsworks connection failed: {e}")

    # Fallback to local CSV
    csv_path = os.path.join(DATA_DIR, "raw_observations.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        return df

    return None


@router.get(
    "/historical",
    summary="Get historical AQI data",
    description="Retrieve historical AQI, weather, and pollutant data for a city.",
)
async def get_historical_data(
    city: str = Query(..., description="City name (karachi, lahore, islamabad)"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(500, ge=1, le=5000, description="Max rows to return"),
    _api_key: str = Depends(verify_api_key),
):
    """
    Get historical AQI data for analytics.

    Returns time series of AQI, weather, and pollutant data.
    """
    try:
        # Load from Hopsworks (primary) or local CSV (fallback)
        df = _load_dataset()
        if df is None:
            return {
                "city": city,
                "count": 0,
                "start": None,
                "end": None,
                "data": [],
                "message": "Historical dataset not available",
            }

        # Filter by city (location_id)
        city_lower = city.lower()
        if "location_id" in df.columns:
            df = df[df["location_id"] == city_lower]
        else:
            return {
                "city": city,
                "count": 0,
                "start": None,
                "end": None,
                "data": [],
                "message": "location_id column not found",
            }

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for city: {city}")

        # Parse timestamps
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Apply date filters
        if start_date:
            start = pd.Timestamp(start_date, tz="UTC")
            df = df[df["timestamp"] >= start]
        if end_date:
            end = pd.Timestamp(end_date, tz="UTC")
            df = df[df["timestamp"] <= end]

        # Sort and limit
        df = df.sort_values("timestamp").tail(limit)

        # Select relevant columns
        display_cols = [
            "timestamp",
            "temperature",
            "humidity",
            "pressure",
            "wind_speed",
            "cloud_cover",
            "precipitation",
            "pm25",
            "pm10",
            "co",
            "no2",
            "so2",
            "o3",
            "aqi",
            "aqi_category",
            "aqi_dominant_pollutant",
        ]
        cols = [c for c in display_cols if c in df.columns]
        result_df = df[cols]

        # Convert to JSON-serializable format
        records = result_df.to_dict(orient="records")
        for r in records:
            if "timestamp" in r:
                r["timestamp"] = r["timestamp"].isoformat() if pd.notna(r["timestamp"]) else None

        return {
            "city": city,
            "count": len(records),
            "start": records[0]["timestamp"] if records else None,
            "end": records[-1]["timestamp"] if records else None,
            "data": records,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Historical data error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@router.get(
    "/statistics",
    summary="Get dataset statistics",
    description="Get summary statistics for a city's data.",
)
async def get_statistics(
    city: str = Query(..., description="City name"),
    _api_key: str = Depends(verify_api_key),
):
    """Get summary statistics for a city."""
    try:
        df = _load_dataset()
        if df is None:
            return {
                "city": city,
                "total_rows": 0,
                "date_range": {"start": None, "end": None},
                "statistics": {},
                "message": "Dataset not available",
            }

        city_lower = city.lower()
        if "location_id" in df.columns:
            df = df[df["location_id"] == city_lower]
        else:
            return {
                "city": city,
                "total_rows": 0,
                "date_range": {"start": None, "end": None},
                "statistics": {},
                "message": "location_id column not found",
            }

        if df.empty:
            return {
                "city": city,
                "total_rows": 0,
                "date_range": {"start": None, "end": None},
                "statistics": {},
                "message": f"No data for city: {city}",
            }

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        numeric_cols = ["aqi", "pm25", "pm10", "temperature", "humidity", "pressure"]
        stats = {}
        for col in numeric_cols:
            if col in df.columns:
                valid = df[col].dropna()
                if len(valid) > 0:
                    stats[col] = {
                        "mean": round(float(valid.mean()), 2),
                        "std": round(float(valid.std()), 2),
                        "min": round(float(valid.min()), 2),
                        "max": round(float(valid.max()), 2),
                        "median": round(float(valid.median()), 2),
                    }

        return {
            "city": city,
            "total_rows": len(df),
            "date_range": {
                "start": str(df["timestamp"].min()),
                "end": str(df["timestamp"].max()),
            },
            "statistics": stats,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@router.get(
    "/compare",
    summary="Compare AQI across cities",
    description="Get AQI time series for all cities for comparison.",
)
async def compare_cities(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    limit: int = Query(500, ge=1, le=5000),
    _api_key: str = Depends(verify_api_key),
):
    """Compare AQI across all cities."""
    try:
        df = _load_dataset()
        if df is None:
            return {
                "data": {"karachi": [], "lahore": [], "islamabad": []},
                "message": "Dataset not available",
            }

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        if start_date:
            df = df[df["timestamp"] >= pd.Timestamp(start_date, tz="UTC")]
        if end_date:
            df = df[df["timestamp"] <= pd.Timestamp(end_date, tz="UTC")]

        # Pivot for comparison
        result = {}
        for city in ["karachi", "lahore", "islamabad"]:
            city_df = df[df["location_id"] == city].sort_values("timestamp").tail(limit)
            city_data = city_df[["timestamp", "aqi", "pm25", "pm10"]].to_dict(orient="records")
            for r in city_data:
                r["timestamp"] = (
                    r["timestamp"].isoformat() if pd.notna(r.get("timestamp")) else None
                )
            result[city] = city_data

        return {"data": result}

    except Exception as e:
        logger.error(f"Compare error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
