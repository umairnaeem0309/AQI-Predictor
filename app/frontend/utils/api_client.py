"""
API Client

Client for communicating with FastAPI backend.
Supports mock mode for development.
"""

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

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
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """
        Initialize API client.

        Args:
            base_url: FastAPI backend URL
            api_key: API key for authentication
            mock_mode: Enable mock mode for development
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for connection errors
            retry_delay: Initial delay between retries (doubles each retry)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.mock_mode = mock_mode
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

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
        
        # Increase timeout for Render (free tier needs wake-up time)
        timeout = int(os.getenv("API_TIMEOUT", "30"))

        return cls(
            base_url=os.getenv("API_BASE_URL", "http://localhost:8000"),
            api_key=api_key,
            mock_mode=mock_mode,
            timeout=timeout,
            max_retries=3,
            retry_delay=3.0,
        )

    def _request_with_retry(self, method: str, url: str, **kwargs) -> requests.Response:
        """
        Make HTTP request with retry logic for connection errors.
        
        Handles Render free-tier sleep/wake by retrying with exponential backoff.
        """
        last_error = None
        delay = self.retry_delay
        
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.request(method, url, timeout=self.timeout, **kwargs)
                response.raise_for_status()
                return response
            except requests.exceptions.ConnectionError as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                    continue
                raise APIConnectionError(
                    f"Cannot connect to API server after {self.max_retries + 1} attempts. "
                    f"Render may be waking up from sleep."
                )
            except requests.exceptions.Timeout as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise APIConnectionError("API request timed out after retries")
            except requests.exceptions.RequestException as e:
                raise APIClientError(f"API error: {e}")

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
            response = self._request_with_retry(
                "POST",
                f"{self.base_url}/prediction",
                json={"city": city},
                headers={"X-API-Key": self.api_key or ""},
            )
            return response.json()
        except APIConnectionError:
            raise
        except APIClientError:
            raise

    def get_health(self) -> Dict[str, Any]:
        """
        Get health status.

        Returns:
            Health response dictionary
        """
        if self.mock_mode:
            return self._mock_health()

        try:
            response = self._request_with_retry(
                "GET",
                f"{self.base_url}/health",
            )
            return response.json()
        except APIConnectionError:
            raise
        except APIClientError:
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Model info response dictionary
        """
        if self.mock_mode:
            return self._mock_model_info()

        try:
            response = self._request_with_retry(
                "GET",
                f"{self.base_url}/model-info",
                headers={"X-API-Key": self.api_key or ""},
            )
            return response.json()
        except APIConnectionError:
            raise
        except APIClientError:
            raise

    def get_historical_data(
        self, city: str, start_date: str = None, end_date: str = None, limit: int = 500
    ) -> Dict[str, Any]:
        """
        Get historical AQI data for a city.

        Args:
            city: City name
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Max rows to return

        Returns:
            Historical data response dictionary
        """
        if self.mock_mode:
            return self._mock_historical(city)

        try:
            params = {"city": city, "limit": limit}
            if start_date:
                params["start_date"] = start_date
            if end_date:
                params["end_date"] = end_date

            response = self._request_with_retry(
                "GET",
                f"{self.base_url}/data/historical",
                params=params,
                headers={"X-API-Key": self.api_key or ""},
            )
            return response.json()
        except APIConnectionError:
            raise
        except APIClientError:
            raise

    def get_statistics(self, city: str) -> Dict[str, Any]:
        """
        Get statistics for a city.

        Args:
            city: City name

        Returns:
            Statistics response dictionary
        """
        if self.mock_mode:
            return self._mock_statistics(city)

        try:
            response = requests.get(
                f"{self.base_url}/data/statistics",
                params={"city": city},
                headers={"X-API-Key": self.api_key or ""},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            raise APIConnectionError("Cannot connect to API server")
        except requests.exceptions.RequestException as e:
            raise APIClientError(f"API error: {e}")

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, Any]:
        """
        Get feature importance from the model.

        Args:
            top_n: Number of top features to return

        Returns:
            Feature importance response dictionary
        """
        if self.mock_mode:
            return self._mock_feature_importance()

        try:
            response = requests.get(
                f"{self.base_url}/explain/feature-importance",
                params={"top_n": top_n},
                headers={"X-API-Key": self.api_key or ""},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            raise APIConnectionError("Cannot connect to API server")
        except requests.exceptions.RequestException as e:
            raise APIClientError(f"API error: {e}")

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get model summary for explainability.

        Returns:
            Model summary response dictionary
        """
        if self.mock_mode:
            return self._mock_model_summary()

        try:
            response = requests.get(
                f"{self.base_url}/explain/model-summary",
                headers={"X-API-Key": self.api_key or ""},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            raise APIConnectionError("Cannot connect to API server")
        except requests.exceptions.RequestException as e:
            raise APIClientError(f"API error: {e}")

    def get_shap_explanation(
        self, features: Dict[str, float], target: str = "target_aqi_24h"
    ) -> Dict[str, Any]:
        """
        Get SHAP explanation for a prediction.

        Args:
            features: Feature values dictionary
            target: Target to explain (e.g. target_aqi_24h)

        Returns:
            SHAP explanation response
        """
        if self.mock_mode:
            return self._mock_shap_explanation()

        try:
            response = requests.post(
                f"{self.base_url}/explain/shap-explanation",
                json={"features": features, "target": target},
                headers={"X-API-Key": self.api_key or ""},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            raise APIConnectionError("Cannot connect to API server")
        except requests.exceptions.RequestException as e:
            raise APIClientError(f"API error: {e}")

    def get_global_shap(self, top_n: int = 20) -> Dict[str, Any]:
        """
        Get global SHAP feature importance.

        Args:
            top_n: Number of top features

        Returns:
            Global SHAP importance response
        """
        if self.mock_mode:
            return self._mock_global_shap()

        try:
            response = requests.get(
                f"{self.base_url}/explain/shap-global",
                params={"top_n": top_n},
                headers={"X-API-Key": self.api_key or ""},
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            raise APIConnectionError("Cannot connect to API server")
        except requests.exceptions.RequestException as e:
            raise APIClientError(f"API error: {e}")

    def get_drift_report(self, n_recent: int = 500) -> Dict[str, Any]:
        """
        Get drift detection report.

        Args:
            n_recent: Number of recent rows to compare

        Returns:
            Drift report response
        """
        if self.mock_mode:
            return {"drift_detected": False, "drifted_count": 0, "drift_percentage": 0}

        try:
            response = requests.get(
                f"{self.base_url}/monitoring/drift",
                params={"n_recent": n_recent},
                headers={"X-API-Key": self.api_key or ""},
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            raise APIConnectionError("Cannot connect to API server")
        except requests.exceptions.RequestException as e:
            raise APIClientError(f"API error: {e}")

    def get_performance(self) -> Dict[str, Any]:
        """
        Get model performance metrics.

        Returns:
            Performance metrics response
        """
        if self.mock_mode:
            return {"status": "healthy", "training_metrics": {}}

        try:
            response = requests.get(
                f"{self.base_url}/monitoring/performance",
                headers={"X-API-Key": self.api_key or ""},
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            raise APIConnectionError("Cannot connect to API server")
        except requests.exceptions.RequestException as e:
            raise APIClientError(f"API error: {e}")

    def get_alerts(self) -> Dict[str, Any]:
        """
        Get AQI hazard alerts.

        Returns:
            Alerts response
        """
        if self.mock_mode:
            return {"alerts": [], "total_alerts": 0}

        try:
            response = requests.get(
                f"{self.base_url}/monitoring/alerts",
                headers={"X-API-Key": self.api_key or ""},
                timeout=15,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            raise APIConnectionError("Cannot connect to API server")
        except requests.exceptions.RequestException as e:
            raise APIClientError(f"API error: {e}")

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get system health overview.

        Returns:
            System health response
        """
        if self.mock_mode:
            return {"overall_status": "healthy", "checks": {}}

        try:
            response = requests.get(
                f"{self.base_url}/monitoring/system-health",
                headers={"X-API-Key": self.api_key or ""},
                timeout=10,
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

    def _mock_historical(self, city: str) -> Dict[str, Any]:
        """Return mock historical data."""
        return {
            "city": city,
            "count": 5,
            "start": "2026-08-20T00:00:00+00:00",
            "end": "2026-08-20T04:00:00+00:00",
            "data": [],
        }

    def _mock_statistics(self, city: str) -> Dict[str, Any]:
        """Return mock statistics."""
        return {
            "city": city,
            "total_rows": 35688,
            "date_range": {"start": "2022-08-01", "end": "2026-08-26"},
            "statistics": {},
        }

    def _mock_feature_importance(self) -> Dict[str, Any]:
        """Return mock feature importance."""
        return {
            "model_name": "xgboost_aqi_predictor",
            "total_features": 71,
            "top_n": 10,
            "features": [],
            "category_importance": {},
        }

    def _mock_model_summary(self) -> Dict[str, Any]:
        """Return mock model summary."""
        return {
            "model_name": "xgboost_aqi_predictor",
            "model_type": "XGBoost",
            "parameters": {},
            "metrics": {"mae": 21.32, "rmse": 30.89, "r2": 0.6065},
        }

    def _mock_shap_explanation(self) -> Dict[str, Any]:
        """Return mock SHAP explanation."""
        return {
            "base_value": 120.0,
            "shap_values": [],
            "feature_names": [],
            "feature_values": [],
            "prediction": 137.0,
            "target": "target_aqi_24h",
            "top_positive": [],
            "top_negative": [],
        }

    def _mock_global_shap(self) -> Dict[str, Any]:
        """Return mock global SHAP."""
        return {
            "model_name": "xgboost_aqi_predictor",
            "method": "TreeExplainer mean |SHAP|",
            "n_samples": 100,
            "total_features": 10,
            "features": [],
        }
