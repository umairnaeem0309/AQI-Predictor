"""
Health Route

Health check endpoint.
"""

import logging

from fastapi import APIRouter

from app.schemas.responses import HealthResponse
from app.services.feature_service import get_feature_service
from app.services.model_service import get_model_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check service health and status.",
)
async def health_check():
    """
    Check service health.

    Returns:
    - **status**: Service status (healthy/unhealthy)
    - **model_loaded**: Whether model is loaded
    - **feature_store_connected**: Feature store connection status
    - **last_prediction**: Timestamp of last prediction
    - **version**: API version
    """
    try:
        model_service = get_model_service()
        feature_service = get_feature_service()

        return HealthResponse(
            status="healthy",
            model_loaded=model_service.is_loaded(),
            feature_store_connected=feature_service.is_connected(),
            last_prediction=None,
            version="1.0.0",
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            feature_store_connected=False,
            last_prediction=None,
            version="1.0.0",
        )


@router.get(
    "/",
    summary="Service availability",
    description="Check if service is available.",
)
async def root():
    """Service availability check."""
    return {"status": "available", "service": "AQI Predictor API"}
