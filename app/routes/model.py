"""
Model Route

Model information endpoint.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from app.backend.dependencies import verify_api_key
from app.schemas.responses import ErrorResponse, ModelInfoResponse
from app.services.model_service import ModelNotLoadedError, get_model_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["model"])


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
    summary="Get model information",
    description="Get production model metadata and metrics.",
)
async def get_model_info(
    request: Request,
    _api_key: str = Depends(verify_api_key),
):
    """
    Get production model information.

    Returns:
    - **model_name**: Model name
    - **model_version**: Model version
    - **status**: Lifecycle status
    - **approval_status**: Approval status
    - **training_date**: Training date
    - **dataset_type**: Dataset type
    - **feature_version**: Feature version
    - **metrics**: Model metrics
    """
    try:
        # Try app.state first (set during lifespan), fallback to global
        model_service = getattr(request.app.state, "model_service", None)
        if model_service is None:
            model_service = get_model_service()
        model_info = model_service.get_model_info()

        return ModelInfoResponse(
            model_name=model_info.get("model_name", "unknown"),
            model_version=model_info.get("model_version", "unknown"),
            status=model_info.get("status", "unknown"),
            approval_status=model_info.get("approval_status", "unknown"),
            training_date=model_info.get("training_date", "unknown"),
            dataset_type=model_info.get("dataset_type", "unknown"),
            feature_version=model_info.get("feature_version", "unknown"),
            metrics=model_info.get("metrics", {}),
        )

    except ModelNotLoadedError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
