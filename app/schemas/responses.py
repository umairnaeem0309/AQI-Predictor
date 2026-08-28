"""
Response Schemas

Pydantic models for API responses.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""

    city: str = Field(..., description="City name")
    timestamp: str = Field(..., description="Prediction timestamp (UTC)")
    aqi_24h: int = Field(..., description="Predicted AQI for 24 hours")
    aqi_48h: int = Field(..., description="Predicted AQI for 48 hours")
    aqi_72h: int = Field(..., description="Predicted AQI for 72 hours")
    category_24h: str = Field(..., description="AQI category for 24h forecast")
    category_48h: str = Field(..., description="AQI category for 48h forecast")
    category_72h: str = Field(..., description="AQI category for 72h forecast")
    model_version: str = Field(..., description="Model version used")
    confidence: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Prediction confidence intervals (level, method, per-horizon bounds)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "city": "Karachi",
                "timestamp": "2026-08-17T10:00:00Z",
                "aqi_24h": 142,
                "aqi_48h": 138,
                "aqi_72h": 145,
                "category_24h": "Unhealthy for Sensitive Groups",
                "category_48h": "Unhealthy for Sensitive Groups",
                "category_72h": "Unhealthy for Sensitive Groups",
                "model_version": "1.0.0",
                "confidence": None,
            }
        }


class FeatureResponse(BaseModel):
    """Response model for feature retrieval endpoint."""

    city: str = Field(..., description="City name")
    timestamp: str = Field(..., description="Feature timestamp (UTC)")
    features: Dict[str, Any] = Field(..., description="Feature values")
    feature_count: int = Field(..., description="Number of features returned")
    feature_version: str = Field(..., description="Feature version")

    class Config:
        json_schema_extra = {
            "example": {
                "city": "Karachi",
                "timestamp": "2026-08-17T10:00:00Z",
                "features": {
                    "temperature": 32.5,
                    "humidity": 65.2,
                    "aqi": 142,
                },
                "feature_count": 3,
                "feature_version": "1.0.0",
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    feature_store_connected: bool = Field(..., description="Feature store connection status")
    last_prediction: Optional[str] = Field(
        default=None,
        description="Timestamp of last prediction",
    )
    version: str = Field(..., description="API version")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "feature_store_connected": True,
                "last_prediction": "2026-08-17T10:00:00Z",
                "version": "1.0.0",
            }
        }


class ModelInfoResponse(BaseModel):
    """Response model for model info endpoint."""

    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    status: str = Field(..., description="Model lifecycle status")
    approval_status: str = Field(..., description="Approval status")
    training_date: str = Field(..., description="Training date")
    dataset_type: str = Field(..., description="Dataset type used for training")
    feature_version: str = Field(..., description="Feature version")
    metrics: Dict[str, float] = Field(..., description="Model metrics")

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "xgboost_v1",
                "model_version": "1.0.0",
                "status": "production",
                "approval_status": "approved",
                "training_date": "2026-08-15",
                "dataset_type": "real_api_data",
                "feature_version": "1.0.0",
                "metrics": {"mae": 15.2, "rmse": 20.1, "r2": 0.85},
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""

    detail: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")

    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Invalid city: Islamabad",
                "type": "InvalidCityError",
            }
        }
