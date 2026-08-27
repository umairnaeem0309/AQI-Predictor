"""
Prediction Service

Handles prediction business logic with monitoring integration.
"""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from src.utils.aqi_categories import get_aqi_category
from src.monitoring.prediction_logger import PredictionLogger
from app.services.model_service import ModelService, ModelNotLoadedError
from app.services.feature_service import FeatureService, FeatureServiceError

logger = logging.getLogger(__name__)


class PredictionServiceError(Exception):
    """Base exception for prediction service."""
    pass


class PredictionError(PredictionServiceError):
    """Prediction computation error."""
    pass


class PredictionService:
    """
    Prediction service for AQI forecasting.
    
    Integrates:
    - Model service for predictions
    - Feature service for input features
    - Prediction logger for monitoring
    """
    
    def __init__(
        self,
        model_service: ModelService,
        feature_service: FeatureService,
        prediction_logger: Optional[PredictionLogger] = None,
    ):
        """
        Initialize prediction service.
        
        Args:
            model_service: Model service instance
            feature_service: Feature service instance
            prediction_logger: Optional prediction logger
        """
        self.model_service = model_service
        self.feature_service = feature_service
        self.prediction_logger = prediction_logger
        self._last_prediction_time: Optional[str] = None
    
    def predict(
        self,
        city: str,
        include_explanation: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate AQI prediction for a city.
        
        Args:
            city: City name
            include_explanation: Include SHAP explanation (future)
            
        Returns:
            Prediction results dictionary
            
        Raises:
            PredictionError: If prediction fails
        """
        start_time = time.time()
        
        try:
            # 1. Validate model is ready
            self.model_service.validate_model_for_request()
            
            # 2. Retrieve features (try feature store, fallback to live adapter)
            try:
                features = self.feature_service.get_features(city)
            except Exception as e:
                logger.warning(f"Feature store unavailable ({e}), using live adapter")
                from src.feature_store.live_feature_adapter import get_live_adapter
                adapter = get_live_adapter()
                features = adapter.get_latest_features(city)
            
            # 3. Run model prediction
            model = self.model_service.get_model()
            predictions = model.predict([list(features.values())])
            
            # 4. Parse predictions
            pred_values = predictions[0] if len(predictions) > 0 else [0, 0, 0]
            
            # 5. Get AQI categories using domain utility
            aqi_24h = int(pred_values[0]) if len(pred_values) > 0 else 0
            aqi_48h = int(pred_values[1]) if len(pred_values) > 1 else 0
            aqi_72h = int(pred_values[2]) if len(pred_values) > 2 else 0
            
            _, category_24h = get_aqi_category(aqi_24h)
            _, category_48h = get_aqi_category(aqi_48h)
            _, category_72h = get_aqi_category(aqi_72h)
            
            # 6. Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # 7. Get model info
            model_info = self.model_service.get_model_info()
            
            # 8. Log prediction (if logger available)
            if self.prediction_logger:
                self.prediction_logger.log_prediction(
                    city=city,
                    model_version=model_info.get("version", "unknown"),
                    input_features=features,
                    predictions={
                        "aqi_24h": aqi_24h,
                        "aqi_48h": aqi_48h,
                        "aqi_72h": aqi_72h,
                    },
                    latency_ms=latency_ms,
                    metadata={"endpoint": "prediction"},
                )
            
            # 9. Update last prediction time
            self._last_prediction_time = datetime.now(timezone.utc).isoformat()
            
            return {
                "city": city,
                "timestamp": self._last_prediction_time,
                "aqi_24h": aqi_24h,
                "aqi_48h": aqi_48h,
                "aqi_72h": aqi_72h,
                "category_24h": category_24h,
                "category_48h": category_48h,
                "category_72h": category_72h,
                "model_version": model_info.get("version", "unknown"),
                "confidence": None,  # Null until uncertainty method implemented
            }
            
        except (ModelNotLoadedError, FeatureServiceError) as e:
            raise PredictionError(f"Prediction failed: {e}")
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise PredictionError(f"Prediction failed: {e}")
    
    def get_last_prediction_time(self) -> Optional[str]:
        """Get timestamp of last prediction."""
        return self._last_prediction_time


# Global prediction service instance
_prediction_service: Optional[PredictionService] = None


def get_prediction_service() -> PredictionService:
    """Get global prediction service instance."""
    global _prediction_service
    if _prediction_service is None:
        raise PredictionServiceError("Prediction service not initialized")
    return _prediction_service


def init_prediction_service(
    model_service: ModelService,
    feature_service: FeatureService,
    prediction_logger: Optional[PredictionLogger] = None,
) -> PredictionService:
    """Initialize global prediction service."""
    global _prediction_service
    _prediction_service = PredictionService(
        model_service=model_service,
        feature_service=feature_service,
        prediction_logger=prediction_logger,
    )
    return _prediction_service
