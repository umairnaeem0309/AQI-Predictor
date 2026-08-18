"""
Prediction Route

Handles prediction requests.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, Request

from app.backend.dependencies import verify_api_key, check_rate_limit
from app.schemas.requests import PredictionRequest, validate_city
from app.schemas.responses import PredictionResponse, ErrorResponse
from app.services.prediction_service import get_prediction_service, PredictionError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/prediction", tags=["prediction"])


@router.post(
    "",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Get 3-day AQI prediction",
    description="Generate AQI prediction for 24h, 48h, and 72h horizons.",
)
async def predict(
    request: PredictionRequest,
    _api_key: str = Depends(verify_api_key),
    _rate_limit: None = Depends(check_rate_limit),
):
    """
    Generate AQI prediction for a city.
    
    - **city**: City name (Karachi, Lahore, Islamabad)
    - **include_explanation**: Include SHAP explanation (future feature)
    """
    try:
        # Validate city
        city = validate_city(request.city)
        
        # Get prediction service
        prediction_service = get_prediction_service()
        
        # Generate prediction
        result = prediction_service.predict(
            city=city,
            include_explanation=request.include_explanation,
        )
        
        return PredictionResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PredictionError as e:
        if "Model not loaded" in str(e):
            raise HTTPException(status_code=503, detail=str(e))
        elif "Feature store" in str(e):
            raise HTTPException(status_code=503, detail=str(e))
        else:
            raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
