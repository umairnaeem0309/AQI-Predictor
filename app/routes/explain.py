"""
Explainability Route

Provides model feature importance, SHAP values, and prediction explanations.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.backend.dependencies import verify_api_key
from app.services.model_service import ModelNotLoadedError, get_model_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/explain", tags=["explain"])


class PredictionExplanationRequest(BaseModel):
    """Request for SHAP prediction explanation."""

    features: Dict[str, float] = Field(..., description="Feature values for the prediction")
    target: str = Field(default="target_aqi_24h", description="Target to explain")


class PredictionExplanationResponse(BaseModel):
    """Response for SHAP prediction explanation."""

    base_value: float
    shap_values: List[Dict[str, Any]]
    feature_names: List[str]
    feature_values: List[float]
    prediction: float
    target: str
    top_positive: List[Dict[str, Any]]
    top_negative: List[Dict[str, Any]]


def _get_feature_names() -> List[str]:
    """Get feature names from model metadata."""
    meta_path = os.path.join("models", "production", "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        return meta.get("feature_columns", [])
    return []


def _get_target_index(target: str) -> int:
    """Map target name to estimator index."""
    meta_path = os.path.join("models", "production", "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        targets = meta.get("target_columns", [])
        if target in targets:
            return targets.index(target)
    # Default to 24h
    return 0


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
        model_info = model_service.get_model_info()

        # Get feature importances - works for both tree and linear models
        if hasattr(model, "estimators_"):
            # MultiOutputRegressor: average importances across targets
            importances = None
            for est in model.estimators_:
                if hasattr(est, "feature_importances_"):
                    # Tree-based model
                    if importances is None:
                        importances = est.feature_importances_.copy()
                    else:
                        importances += est.feature_importances_
                elif hasattr(est, "coef_"):
                    # Linear model (Ridge): use absolute coefficients
                    coefs = np.abs(est.coef_)
                    if importances is None:
                        importances = coefs.copy()
                    else:
                        importances += coefs
            if importances is not None:
                importances /= len(model.estimators_)
        elif hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_)
        else:
            raise HTTPException(status_code=500, detail="Model does not support feature importance")

        feature_names = _get_feature_names()
        if not feature_names:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        # Create sorted importance list
        importance_list = []
        for name, score in zip(feature_names, importances):
            importance_list.append({"feature": name, "importance": round(float(score), 4)})

        importance_list.sort(key=lambda x: x["importance"], reverse=True)
        top_features = importance_list[:top_n]

        # Categorize features
        categories = {
            "weather": [
                "temperature",
                "humidity",
                "pressure",
                "wind_speed",
                "wind_direction",
                "cloud_cover",
                "precipitation",
            ],
            "pollution": ["pm25", "pm10", "co", "no2", "so2", "o3"],
            "lag": [f for f in feature_names if "lag" in f],
            "rolling": [f for f in feature_names if "rolling" in f],
            "time": [
                "hour",
                "day_of_week",
                "month",
                "is_weekend",
                "season",
                "hour_sin",
                "hour_cos",
            ],
            "interaction": [
                f
                for f in feature_names
                if "ratio" in f
                or "interaction" in f
                or "deviation" in f
                or "change_rate" in f
                or "trend" in f
                or "cooling" in f
            ],
        }

        # Count importance by category
        category_importance = {}
        for cat, cat_features in categories.items():
            cat_total = sum(
                imp["importance"] for imp in importance_list if imp["feature"] in cat_features
            )
            category_importance[cat] = round(cat_total, 4)

        return {
            "model_name": model_info.get("model_name", "unknown"),
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
            "model_name": model_info.get("model_name", "unknown"),
            "model_type": model_info.get("model_key", "unknown") + " (MultiOutputRegressor)",
            "parameters": meta.get("model_params", {}),
            "metrics": model_info.get("metrics", {}),
            "val_metrics": model_info.get("val_metrics", {}),
            "test_metrics": model_info.get("test_metrics", {}),
            "model_comparison": model_info.get("model_comparison", {}),
            "feature_count": len(meta.get("feature_columns", [])),
            "target_count": len(meta.get("target_columns", [])),
            "targets": meta.get("target_columns", []),
            "training_data": {
                "provider": model_info.get("data_provider", "open-meteo"),
                "date_range": "2022-08 to 2024-12",
                "cities": ["karachi", "lahore", "islamabad"],
                "total_hours": 107064,
                "train_rows": meta.get("train_rows", 0),
                "val_rows": meta.get("val_rows", 0),
                "test_rows": meta.get("test_rows", 0),
            },
            "aqi_method": "US EPA PM NowCast AQI (EPA-454/B-24-002, May 2024)",
            "data_source": "Open-Meteo Historical Weather + Air Quality APIs",
        }

    except ModelNotLoadedError:
        raise HTTPException(status_code=503, detail="Model not loaded")
    except Exception as e:
        logger.error(f"Model summary error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@router.post(
    "/shap-explanation",
    summary="Get SHAP explanation for a prediction",
    description="Compute SHAP values for a given feature vector to explain model predictions.",
    response_model=PredictionExplanationResponse,
)
async def get_shap_explanation(
    request: PredictionExplanationRequest,
    _api_key: str = Depends(verify_api_key),
):
    """
    Compute SHAP TreeExplainer values for a prediction.

    This uses the XGBoost tree structure for exact SHAP computation,
    providing feature-level contributions to the prediction.
    """
    try:
        import shap

        model_service = get_model_service()
        model = model_service.get_model()

        feature_names = _get_feature_names()
        if not feature_names:
            raise HTTPException(status_code=500, detail="Feature names not found in metadata")

        # Compute training means for filling missing features
        import pandas as pd

        train_means = {}
        # Try Hopsworks first
        try:
            from src.feature_store import get_feature_store

            store = get_feature_store()
            for fg_name in ["aqi_features_prod", "aqi_features_test"]:
                try:
                    df_means = store.get_features(fg_name, version=1)
                    if not df_means.empty:
                        train_means = df_means.select_dtypes(include=[np.number]).mean().to_dict()
                        break
                except Exception:
                    continue
        except Exception:
            pass

        # Fallback to local CSV
        if not train_means:
            for candidate in [
                os.path.join("data", "processed", "train_features.csv"),
                os.path.join("data", "processed", "raw_observations.csv"),
            ]:
                if os.path.exists(candidate):
                    try:
                        df_means = pd.read_csv(candidate, nrows=1000)
                        train_means = df_means.select_dtypes(include=[np.number]).mean().to_dict()
                    except Exception:
                        pass
                    break

        # Build feature vector in correct order, fill missing with training mean
        feature_values = []
        missing_count = 0
        for fname in feature_names:
            if fname in request.features:
                feature_values.append(request.features[fname])
            else:
                # Use training mean, fallback to 0
                feature_values.append(float(train_means.get(fname, 0.0)))
                missing_count += 1

        X = np.array([feature_values], dtype=np.float64)

        # Get the correct estimator for the requested target
        target_idx = _get_target_index(request.target)

        if hasattr(model, "estimators_"):
            estimator = model.estimators_[target_idx]
        else:
            estimator = model

        # Compute SHAP values using appropriate explainer
        if hasattr(estimator, "feature_importances_"):
            # Tree-based model (XGBoost, Random Forest)
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X)
        else:
            # Linear model (Ridge) or other
            try:
                explainer = shap.LinearExplainer(estimator, np.zeros((10, X.shape[1])))
                shap_values = explainer.shap_values(X)
            except Exception:
                # Fallback: use KernelExplainer with a small background
                background = np.zeros((5, X.shape[1]))
                explainer = shap.KernelExplainer(estimator.predict, background)
                shap_values = explainer.shap_values(X, nsamples=100)

        # shap_values shape: (1, n_features) or (n_features,)
        if isinstance(shap_values, np.ndarray) and shap_values.ndim > 1:
            sv = shap_values[0]
        elif isinstance(shap_values, np.ndarray):
            sv = shap_values
        else:
            sv = np.array(shap_values)

        # Get base value (expected value)
        try:
            ev = explainer.expected_value
            if isinstance(ev, (list, np.ndarray)):
                base_value = float(ev[0]) if len(ev) > 0 else 0.0
            else:
                base_value = float(ev)
        except Exception:
            base_value = 0.0

        prediction = float(base_value + np.sum(sv))

        # Build per-feature SHAP list
        shap_list = []
        for fname, val, sv_val in zip(feature_names, feature_values, sv):
            shap_list.append(
                {
                    "feature": fname,
                    "shap_value": round(float(sv_val), 4),
                    "feature_value": round(float(val), 4),
                }
            )

        # Sort by absolute SHAP value
        shap_list.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        # Split into positive and negative
        top_positive = [
            {
                "feature": s["feature"],
                "shap_value": s["shap_value"],
                "feature_value": s["feature_value"],
            }
            for s in shap_list
            if s["shap_value"] > 0
        ][:10]

        top_negative = [
            {
                "feature": s["feature"],
                "shap_value": s["shap_value"],
                "feature_value": s["feature_value"],
            }
            for s in shap_list
            if s["shap_value"] < 0
        ][:10]

        return PredictionExplanationResponse(
            base_value=round(base_value, 4),
            shap_values=shap_list[:20],
            feature_names=feature_names,
            feature_values=[round(float(v), 4) for v in feature_values],
            prediction=round(prediction, 4),
            target=request.target,
            top_positive=top_positive,
            top_negative=top_negative,
        )

    except ImportError:
        raise HTTPException(status_code=500, detail="SHAP library not installed")
    except ModelNotLoadedError:
        raise HTTPException(status_code=503, detail="Model not loaded")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SHAP explanation error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@router.get(
    "/shap-global",
    summary="Get global SHAP feature importance",
    description="Compute global SHAP importance using a background sample from training data.",
)
async def get_global_shap_importance(
    top_n: int = 20,
    n_samples: int = 100,
    _api_key: str = Depends(verify_api_key),
):
    """
    Compute global SHAP importance by averaging |SHAP| values
    over a sample of the training data.

    This gives a more faithful importance ranking than XGBoost's
    built-in feature_importances_ (which uses split-based gain).
    """
    try:
        import pandas as pd
        import shap

        model_service = get_model_service()
        model = model_service.get_model()
        model_info = model_service.get_model_info()

        feature_names = _get_feature_names()
        if not feature_names:
            raise HTTPException(status_code=500, detail="Feature names not found in metadata")

        # Load feature-engineered dataset for background sample
        # Try Hopsworks first, then local CSV files
        df = None
        try:
            from src.feature_store import get_feature_store

            store = get_feature_store()
            for fg_name in ["aqi_features_prod", "aqi_features_test"]:
                try:
                    df = store.get_features(fg_name, version=1)
                    if not df.empty:
                        break
                except Exception:
                    continue
        except Exception:
            pass

        if df is None or df.empty:
            data_path = os.path.join("data", "processed", "train_features.csv")
            if not os.path.exists(data_path):
                data_path = os.path.join("data", "processed", "raw_observations.csv")
            if not os.path.exists(data_path):
                return {
                    "model_name": model_info.get("model_name", "unknown"),
                    "method": "LinearExplainer" if not hasattr(model.estimators_[0], "feature_importances_") else "TreeExplainer",
                    "n_samples": 0,
                    "total_features": 0,
                    "features": [],
                    "message": "Training dataset not available for SHAP background sample.",
                }
            df = pd.read_csv(data_path)

        # Select only the feature columns that exist in the data
        available = [f for f in feature_names if f in df.columns]
        if len(available) < len(feature_names) * 0.5:
            raise HTTPException(status_code=500, detail="Insufficient feature columns in dataset")

        # Sample for efficiency
        sample_size = min(n_samples, len(df))
        bg_sample = df[available].sample(n=sample_size, random_state=42).values

        # Pad missing features with 0
        if len(available) < len(feature_names):
            padding = np.zeros((sample_size, len(feature_names) - len(available)))
            bg_sample = np.hstack([bg_sample, padding])

        X_bg = bg_sample.astype(np.float64)

        # Get target estimator (default 24h)
        target_idx = 0
        if hasattr(model, "estimators_"):
            estimator = model.estimators_[target_idx]
        else:
            estimator = model

        # Use appropriate explainer
        if hasattr(estimator, "feature_importances_"):
            explainer = shap.TreeExplainer(estimator, data=X_bg)
            shap_values = explainer.shap_values(X_bg)
        else:
            try:
                explainer = shap.LinearExplainer(estimator, X_bg)
                shap_values = explainer.shap_values(X_bg)
            except Exception:
                explainer = shap.KernelExplainer(estimator.predict, X_bg[:10])
                shap_values = explainer.shap_values(X_bg[:20], nsamples=50)

        if isinstance(shap_values, np.ndarray) and shap_values.ndim > 1:
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        elif isinstance(shap_values, np.ndarray):
            mean_abs_shap = np.abs(shap_values)
        else:
            mean_abs_shap = np.abs(np.array(shap_values))

        # Build sorted list
        importance_list = []
        for fname, val in zip(feature_names, mean_abs_shap):
            importance_list.append({"feature": fname, "mean_abs_shap": round(float(val), 4)})

        importance_list.sort(key=lambda x: x["mean_abs_shap"], reverse=True)

        return {
            "model_name": model_info.get("model_name", "unknown"),
            "method": "TreeExplainer mean |SHAP|",
            "n_samples": sample_size,
            "total_features": len(importance_list),
            "features": importance_list[:top_n],
        }

    except ImportError:
        raise HTTPException(status_code=500, detail="SHAP library not installed")
    except ModelNotLoadedError:
        raise HTTPException(status_code=503, detail="Model not loaded")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Global SHAP error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
