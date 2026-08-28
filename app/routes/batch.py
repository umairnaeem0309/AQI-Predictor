"""
Batch Prediction Route

Provides batch prediction endpoint for multiple cities.
"""

import logging
import time
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.backend.dependencies import check_rate_limit, verify_api_key
from app.schemas.requests import validate_city
from app.services.model_service import ModelNotLoadedError
from app.services.prediction_service import PredictionError, get_prediction_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/batch", tags=["batch"])


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""

    cities: List[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="List of city names (max 10)",
    )


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""

    predictions: list
    total_cities: int
    successful: int
    failed: int
    total_time_ms: float


@router.post(
    "/predictions",
    response_model=BatchPredictionResponse,
    summary="Get predictions for multiple cities",
    description="Generate AQI predictions for a list of cities in a single request.",
)
async def batch_predict(
    request: BatchPredictionRequest,
    _api_key: str = Depends(verify_api_key),
    _rate_limit: None = Depends(check_rate_limit),
):
    """
    Generate AQI predictions for multiple cities.

    - **cities**: List of city names (max 10 per request)
    """
    start_time = time.time()
    prediction_service = get_prediction_service()

    predictions = []
    successful = 0
    failed = 0

    for city_name in request.cities:
        try:
            city = validate_city(city_name)
            result = prediction_service.predict(city=city)
            predictions.append(result)
            successful += 1
        except ValueError as e:
            predictions.append(
                {
                    "city": city_name,
                    "error": str(e),
                    "aqi_24h": None,
                }
            )
            failed += 1
        except PredictionError as e:
            predictions.append(
                {
                    "city": city_name,
                    "error": str(e),
                    "aqi_24h": None,
                }
            )
            failed += 1
        except ModelNotLoadedError as e:
            predictions.append(
                {
                    "city": city_name,
                    "error": "Model not loaded",
                    "aqi_24h": None,
                }
            )
            failed += 1
        except Exception as e:
            logger.error(f"Batch prediction error for {city_name}: {e}")
            predictions.append(
                {
                    "city": city_name,
                    "error": "Internal error",
                    "aqi_24h": None,
                }
            )
            failed += 1

    elapsed_ms = (time.time() - start_time) * 1000

    return BatchPredictionResponse(
        predictions=predictions,
        total_cities=len(request.cities),
        successful=successful,
        failed=failed,
        total_time_ms=round(elapsed_ms, 2),
    )
