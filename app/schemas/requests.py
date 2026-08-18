"""
Request Schemas

Pydantic models for API request validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    
    city: str = Field(
        ...,
        description="City name for prediction",
        examples=["Karachi", "Lahore", "Islamabad"],
    )
    include_explanation: bool = Field(
        default=False,
        description="Include SHAP explanation (future feature)",
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "city": "Karachi",
                "include_explanation": False,
            }
        }


class FeatureRequest(BaseModel):
    """Request model for feature retrieval endpoint."""
    
    city: str = Field(
        ...,
        description="City name",
        examples=["Karachi", "Lahore", "Islamabad"],
    )
    feature_names: Optional[List[str]] = Field(
        default=None,
        description="Specific feature names to retrieve (all if null)",
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "city": "Karachi",
                "feature_names": ["temperature", "humidity", "aqi"],
            }
        }


# Valid cities
VALID_CITIES = {"karachi", "lahore", "islamabad"}


def validate_city(city: str) -> str:
    """
    Validate city name.
    
    Args:
        city: City name to validate
        
    Returns:
        Normalized city name (lowercase)
        
    Raises:
        ValueError: If city is not valid
    """
    normalized = city.lower().strip()
    if normalized not in VALID_CITIES:
        raise ValueError(
            f"Invalid city: {city}. Valid cities: {', '.join(sorted(VALID_CITIES))}"
        )
    return normalized
