"""
Explainability Route

Provides model feature importance and prediction explanations.
"""

import logging
import os
import json
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException

from app.backend.dependencies import verify_api_key
from app.services.model_service import get_model_service, ModelNotLoadedError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/explain", tags=["explain"])


@router.get(
    "/feature-importance",
    summary="Get feature importance",
    description="Get top N most important features from the XGBoost model.",
)
async def get_feature_importance(
    top_n: int = 20,
    _api_key: str = Depends(verify_api_key),
):
    """
    Get feature importance from the trained XGBoost model.

    Returns feature names, importance scores, and categories.
    """
    try:
        model_service = get_model_service()
        model = model_service.get_model()

        # Get feature importances from XGBoost
        # Model may be wrapped in MultiOutputRegressor
        if hasattr(model, 'estimators_'):
            # MultiOutputRegressor: average importances across targets
            importances = None
            for est in model.estimators_:
                if hasattr(est, 'feature_importances_'):
                    if importances is None:
                        importances = est.feature_importances_.copy()
                    else:
                        importances += est.feature_importances_
            if importances is not None:
                importances /= len(model.estimators_)
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            raise HTTPException(status_code=500, detail="Model does not support feature importance")

        # Get feature names from metadata
        meta_path = os.path.join("models", "production", "model_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            feature_names = meta.get("feature_columns", [f"feature_{i}" for i in range(len(importances))])
        else:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        # Create sorted importance list
        importance_list = []
        for name, score in zip(feature_names, importances):
            importance_list.append({"feature": name, "importance": round(float(score), 4)})

        importance_list.sort(key=lambda x: x["importance"], reverse=True)
        top_features = importance_list[:top_n]

        # Categorize features
        categories = {
            "weather": ["temperature", "humidity", "pressure", "wind_speed", "wind_direction", "cloud_cover", "precipitation"],
            "pollution": ["pm25", "pm10", "co", "no2", "so2", "o3"],
            "lag": [f for f in feature_names if "lag" in f],
            "rolling": [f for f in feature_names if "rolling" in f],
            "time": ["hour", "day_of_week", "month", "is_weekend", "season", "hour_sin", "hour_cos"],
            "interaction": [f for f in feature_names if "ratio" in f or "interaction" in f or "deviation" in f or "change_rate" in f or "trend" in f or "cooling" in f],
        }

        # Count importance by category
        category_importance = {}
        for cat, cat_features in categories.items():
            cat_total = sum(
                imp["importance"]
                for imp in importance_list
                if imp["feature"] in cat_features
            )
            category_importance[cat] = round(cat_total, 4)

        return {
            "model_name": "xgboost_aqi_predictor",
            "total_features": len(importance_list),
            "top_n": top_n,
            "features": top_features,
            "category_importance": category_importance,
        }

    except ModelNotLoadedError:
        raise HTTPException(status_code=503, detail="Model not loaded")
    except Exception as e:
        logger.error(f"Feature importance error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@router.get(
    "/model-summary",
    summary="Get model summary",
    description="Get model architecture, parameters, and performance summary.",
)
async def get_model_summary(
    _api_key: str = Depends(verify_api_key),
):
    """Get a comprehensive model summary for explainability."""
    try:
        model_service = get_model_service()
        model_info = model_service.get_model_info()

        meta_path = os.path.join("models", "production", "model_metadata.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)

        return {
            "model_name": model_info.get("model_name", "xgboost_aqi_predictor"),
            "model_type": "XGBoost (MultiOutputRegressor)",
            "parameters": meta.get("model_params", {}),
            "metrics": model_info.get("metrics", {}),
            "feature_count": len(meta.get("feature_columns", [])),
            "target_count": len(meta.get("target_columns", [])),
            "targets": meta.get("target_columns", []),
            "training_data": {
                "provider": model_info.get("data_provider", "open-meteo"),
                "date_range": "2022-08 to 2026-08",
                "cities": ["karachi", "lahore", "islamabad"],
                "total_hours": 107064,
            },
            "aqi_method": "US EPA PM NowCast AQI (EPA-454/B-24-002, May 2024)",
            "data_source": "Open-Meteo Historical Weather + Air Quality APIs",
        }

    except ModelNotLoadedError:
        raise HTTPException(status_code=503, detail="Model not loaded")
    except Exception as e:
        logger.error(f"Model summary error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
