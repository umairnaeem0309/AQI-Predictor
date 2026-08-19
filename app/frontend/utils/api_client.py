"""
API Client

Client for communicating with FastAPI backend.
Supports mock mode for development.
"""

import os
import time
from typing import Optional, Dict, Any
from datetime import datetime, timezone

import requests


class APIClientError(Exception):
    """API client error."""
    pass


class APIConnectionError(APIClientError):
    """API connection error."""
    pass


class APIClient:
    """
    API client for AQI Predictor backend.
    
    Modes:
    - production: Calls real FastAPI backend
    - mock: Returns mock data for development
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        mock_mode: bool = False,
        timeout: int = 10,
    ):
        """
        Initialize API client.
        
        Args:
            base_url: FastAPI backend URL
            api_key: API key for authentication
            mock_mode: Enable mock mode for development
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.mock_mode = mock_mode
        self.timeout = timeout
    
    @classmethod
    def from_env(cls) -> "APIClient":
        """Create client from environment variables."""
        # Check for Streamlit secrets first
        try:
            import streamlit as st
            api_key = st.secrets.get("API_KEY", os.getenv("API_KEY"))
        except (ImportError, FileNotFoundError):
            api_key = os.getenv("API_KEY")
        
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        
        return cls(
            base_url=os.getenv("API_BASE_URL", "http://localhost:8000"),
            api_key=api_key,
            mock_mode=mock_mode,
        )
    
    def get_prediction(self, city: str) -> Dict[str, Any]:
        """
        Get prediction for a city.
        
        Args:
            city: City name
            
        Returns:
            Prediction response dictionary
        """
        if self.mock_mode:
            return self._mock_prediction(city)
        
        try:
            response = requests.post(
                f"{self.base_url}/prediction",
                json={"city": city},
                headers={"X-API-Key": self.api_key or ""},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            raise APIConnectionError("Cannot connect to API server")
        except requests.exceptions.Timeout:
            raise APIConnectionError("API request timed out")
        except requests.exceptions.RequestException as e:
            raise APIClientError(f"API error: {e}")
    
    def get_health(self) -> Dict[str, Any]:
        """
        Get health status.
        
        Returns:
            Health response dictionary
        """
        if self.mock_mode:
            return self._mock_health()
        
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            raise APIConnectionError("Cannot connect to API server")
        except requests.exceptions.RequestException as e:
            raise APIClientError(f"API error: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Model info response dictionary
        """
        if self.mock_mode:
            return self._mock_model_info()
        
        try:
            response = requests.get(
                f"{self.base_url}/model-info",
                headers={"X-API-Key": self.api_key or ""},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            raise APIConnectionError("Cannot connect to API server")
        except requests.exceptions.RequestException as e:
            raise APIClientError(f"API error: {e}")
    
    def is_available(self) -> bool:
        """Check if API is available."""
        if self.mock_mode:
            return True
        
        try:
            self.get_health()
            return True
        except APIClientError:
            return False
    
    def _mock_prediction(self, city: str) -> Dict[str, Any]:
        """Return mock prediction data."""
        import random
        base_aqi = {"Karachi": 140, "Lahore": 160, "Islamabad": 90}
        base = base_aqi.get(city, 120)
        
        return {
            "city": city,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "aqi_24h": base + random.randint(-20, 20),
            "aqi_48h": base + random.randint(-25, 25),
            "aqi_72h": base + random.randint(-30, 30),
            "category_24h": "Unhealthy for Sensitive Groups",
            "category_48h": "Unhealthy for Sensitive Groups",
            "category_72h": "Unhealthy for Sensitive Groups",
            "model_version": "mock-v1.0.0",
            "confidence": None,
        }
    
    def _mock_health(self) -> Dict[str, Any]:
        """Return mock health data."""
        return {
            "status": "healthy",
            "model_loaded": True,
            "feature_store_connected": True,
            "last_prediction": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0",
        }
    
    def _mock_model_info(self) -> Dict[str, Any]:
        """Return mock model info."""
        return {
            "model_name": "mock-model",
            "model_version": "mock-v1.0.0",
            "status": "production",
            "approval_status": "approved",
            "training_date": "2026-08-15",
            "dataset_type": "real_api_data",
            "feature_version": "1.0.0",
            "metrics": {"mae": 15.2, "rmse": 20.1, "r2": 0.85},
        }
