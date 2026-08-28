"""
Monitoring Route

Provides data drift detection, performance monitoring, and system health endpoints.
"""

import logging
import os
import json
from typing import Dict, Any, Optional
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from app.backend.dependencies import verify_api_key
from app.services.model_service import get_model_service, ModelNotLoadedError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

# AQI categories for alerting
AQI_CATEGORIES = {
    "Good": (0, 50),
    "Moderate": (51, 100),
    "Unhealthy for Sensitive Groups": (101, 150),
    "Unhealthy": (151, 200),
    "Very Unhealthy": (201, 300),
    "Hazardous": (301, 500),
}


def _get_aqi_category(aqi: float) -> str:
    """Get AQI category from value."""
    for cat, (low, high) in AQI_CATEGORIES.items():
        if low <= aqi <= high:
            return cat
    return "Hazardous" if aqi > 300 else "Good"


def _load_processed_data() -> Optional[pd.DataFrame]:
    """Load processed dataset for drift analysis."""
    # Prefer feature-engineered dataset (has all 71 model features)
    candidates = [
        os.path.join("data", "processed", "train_features.csv"),
        os.path.join("data", "processed", "raw_observations.csv"),
    ]
    for data_path in candidates:
        if os.path.exists(data_path):
            try:
                return pd.read_csv(data_path)
            except Exception as e:
                logger.warning(f"Failed to load {data_path}: {e}")
    return None


@router.get(
    "/drift",
    summary="Run drift detection",
    description="Compare recent data against training baseline to detect drift.",
)
async def get_drift_report(
    n_recent: int = Query(default=500, description="Number of recent rows to compare"),
    _api_key: str = Depends(verify_api_key),
):
    """
    Run data drift detection using Evidently AI.

    Compares the most recent n_recent observations against the full training dataset.
    """
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        df = _load_processed_data()
        if df is None or df.empty:
            return {
                "status": "unavailable",
                "message": "Training dataset not available in this environment. Drift detection requires train_features.csv or raw_observations.csv.",
                "drift_detected": False,
                "drifted_count": 0,
                "drift_percentage": 0,
                "total_features": 0,
            }

        # Select numeric columns for drift detection
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove targets and ID-like columns
        exclude = [c for c in numeric_cols if c.startswith("target_") or c == "location_id"]
        feature_cols = [c for c in numeric_cols if c not in exclude]

        if len(feature_cols) < 3:
            raise HTTPException(status_code=400, detail="Insufficient numeric features for drift analysis")

        # Split: reference = first 80%, current = last 20%
        split_idx = int(len(df) * 0.8)
        reference = df[feature_cols].iloc[:split_idx].dropna()
        current = df[feature_cols].iloc[split_idx:].dropna()

        if len(reference) < 100 or len(current) < 50:
            raise HTTPException(status_code=400, detail="Insufficient data for drift analysis")

        # Limit current to n_recent
        current = current.tail(n_recent)

        # Run Evidently
        report = Report([DataDriftPreset(method="psi")])
        evaluation = report.run(current, reference)
        report_dict = evaluation.dict()

        # Parse results
        drifted_columns = []
        total_columns = len(feature_cols)
        drift_count = 0

        for metric in report_dict.get("metrics", []):
            metric_name = metric.get("metric_name", "")
            if "DriftedColumnsCount" in metric_name:
                value = metric.get("value", {})
                if isinstance(value, dict):
                    drift_count = value.get("count", 0)
                    total_columns = value.get("drifted", 0) + value.get("not_drifted", 0)

            if metric_name.startswith("ValueDrift"):
                config = metric.get("config", {})
                col_name = config.get("column", "unknown")
                score = metric.get("value", 0.0)
                if isinstance(score, bool) and score:
                    drifted_columns.append({"column": col_name, "drifted": True})
                elif isinstance(score, (int, float)) and score > 0.1:
                    drifted_columns.append({"column": col_name, "drifted": True, "score": float(score)})

        drift_percentage = (len(drifted_columns) / total_columns * 100) if total_columns > 0 else 0

        return {
            "status": "completed",
            "reference_rows": len(reference),
            "current_rows": len(current),
            "total_features": total_columns,
            "drifted_count": drift_count,
            "drift_percentage": round(drift_percentage, 2),
            "drifted_columns": drifted_columns,
            "drift_detected": drift_count > 0,
            "threshold": "PSI > 0.1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    except ImportError:
        raise HTTPException(status_code=500, detail="Evidently library not installed")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Drift detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Drift detection failed: {e}")


@router.get(
    "/performance",
    summary="Get model performance metrics",
    description="Get current model performance metrics from training.",
)
async def get_performance_metrics(
    _api_key: str = Depends(verify_api_key),
):
    """Get model performance metrics."""
    try:
        meta_path = os.path.join("models", "production", "model_metadata.json")
        if not os.path.exists(meta_path):
            raise HTTPException(status_code=404, detail="Model metadata not found")

        with open(meta_path) as f:
            meta = json.load(f)

        metrics = meta.get("metrics", {})

        return {
            "status": "healthy",
            "training_metrics": metrics,
            "model_version": meta.get("model_version", "v1.0.0"),
            "training_date": meta.get("training_date", "unknown"),
            "data_provider": meta.get("data_provider", "open-meteo"),
            "feature_count": len(meta.get("feature_columns", [])),
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Performance metrics error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@router.get(
    "/alerts",
    summary="Get AQI alerts",
    description="Check current AQI predictions against hazard thresholds.",
)
async def get_alerts(
    _api_key: str = Depends(verify_api_key),
):
    """
    Check current predictions for hazardous AQI levels.

    Returns alerts for any city where predicted AQI exceeds thresholds.
    """
    try:
        from src.data.providers.open_meteo_air_quality import OpenMeteoAirQualityProvider

        alerts = []
        cities = {
            "karachi": {"lat": 24.8607, "lon": 67.0011},
            "lahore": {"lat": 31.5204, "lon": 74.3587},
            "islamabad": {"lat": 33.6844, "lon": 73.0479},
        }

        provider = OpenMeteoAirQualityProvider()

        for city, coords in cities.items():
            try:
                data = provider.fetch_current(
                    latitude=coords["lat"],
                    longitude=coords["lon"],
                )
                if data and "current" in data:
                    current = data["current"]
                    pm25 = current.get("pm2_5", 0)
                    pm10 = current.get("pm10", 0)

                    # Simple AQI estimation from PM2.5
                    if pm25 <= 9.0:
                        aqi = int(pm25 * 50 / 9.0)
                    elif pm25 <= 35.4:
                        aqi = int(50 + (pm25 - 9.0) * 49 / 26.4)
                    elif pm25 <= 55.4:
                        aqi = int(100 + (pm25 - 35.4) * 49 / 20.0)
                    elif pm25 <= 150.4:
                        aqi = int(150 + (pm25 - 55.4) * 49 / 95.0)
                    elif pm25 <= 250.4:
                        aqi = int(200 + (pm25 - 150.4) * 99 / 100.0)
                    else:
                        aqi = int(300 + (pm25 - 250.4) * 199 / 249.6)

                    category = _get_aqi_category(aqi)

                    alert_level = "none"
                    if aqi > 200:
                        alert_level = "critical"
                    elif aqi > 150:
                        alert_level = "warning"
                    elif aqi > 100:
                        alert_level = "caution"

                    if alert_level != "none":
                        alerts.append({
                            "city": city.title(),
                            "aqi": aqi,
                            "pm25": pm25,
                            "pm10": pm10,
                            "category": category,
                            "alert_level": alert_level,
                            "recommendation": _get_recommendation(aqi),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })

            except Exception as e:
                logger.warning(f"Failed to fetch data for {city}: {e}")

        return {
            "alerts": alerts,
            "total_alerts": len(alerts),
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Alert check error: {e}")
        raise HTTPException(status_code=500, detail=f"Alert check failed: {e}")


def _get_recommendation(aqi: int) -> str:
    """Get health recommendation for AQI level."""
    if aqi <= 50:
        return "Air quality is satisfactory. No risk."
    elif aqi <= 100:
        return "Acceptable. Unusually sensitive people should consider reducing prolonged outdoor exertion."
    elif aqi <= 150:
        return "Members of sensitive groups may experience health effects. Limit prolonged outdoor exertion."
    elif aqi <= 200:
        return "Everyone may begin to experience health effects. Avoid prolonged outdoor exertion."
    elif aqi <= 300:
        return "Health alert: everyone may experience serious health effects. Avoid all outdoor exertion."
    else:
        return "Health emergency: everyone should avoid all outdoor physical activity."


@router.get(
    "/system-health",
    summary="Get system health overview",
    description="Get comprehensive system health status.",
)
async def get_system_health(
    _api_key: str = Depends(verify_api_key),
):
    """Get comprehensive system health including all components."""
    checks = {}

    # Model check
    try:
        model_service = get_model_service()
        model_service.validate_model_for_request()
        checks["model"] = {"status": "healthy", "loaded": True}
    except Exception:
        checks["model"] = {"status": "unhealthy", "loaded": False}

    # Data check
    df = _load_processed_data()
    if df is not None:
        checks["dataset"] = {
            "status": "healthy",
            "rows": len(df),
            "columns": len(df.columns),
            "cities": df["city"].nunique() if "city" in df.columns else 0,
        }
    else:
        checks["dataset"] = {"status": "unavailable"}

    # API check
    checks["api"] = {"status": "healthy"}

    overall = all(c.get("status") == "healthy" for c in checks.values())

    return {
        "overall_status": "healthy" if overall else "degraded",
        "checks": checks,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
