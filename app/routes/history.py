"""
Prediction History Route

Provides endpoints for querying stored prediction history.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.backend.dependencies import verify_api_key
from src.data.prediction_history import PredictionHistoryStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/history", tags=["history"])

# Default store path
_DEFAULT_DB = "data/predictions/prediction_history.db"


def _get_store() -> PredictionHistoryStore:
    """Get prediction history store instance."""
    return PredictionHistoryStore(db_path=_DEFAULT_DB)


@router.get(
    "/predictions",
    summary="Get prediction history",
    description="Query past predictions stored in the history database.",
)
async def get_prediction_history(
    city: Optional[str] = Query(default=None, description="Filter by city"),
    start_date: Optional[str] = Query(default=None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(default=None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(default=100, ge=1, le=1000, description="Max results"),
    _api_key: str = Depends(verify_api_key),
):
    """
    Query prediction history from SQLite storage.

    Supports filtering by city, date range, and limit.
    """
    store = _get_store()
    predictions = store.get_history(
        city=city,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )

    return {
        "count": len(predictions),
        "predictions": predictions,
    }


@router.get(
    "/stats",
    summary="Get prediction statistics",
    description="Get statistics about stored predictions.",
)
async def get_prediction_stats(
    city: Optional[str] = Query(default=None, description="Filter by city"),
    _api_key: str = Depends(verify_api_key),
):
    """Get aggregated prediction statistics."""
    store = _get_store()
    stats = store.get_stats(city=city)
    return stats
