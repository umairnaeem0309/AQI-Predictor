"""
Feature Service

Handles feature retrieval from feature store.
Hopsworks primary, local fallback only when explicitly enabled.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from src.feature_store.base import FeatureStoreInterface

logger = logging.getLogger(__name__)


class FeatureServiceError(Exception):
    """Base exception for feature service."""
    pass


class FeatureStoreConnectionError(FeatureServiceError):
    """Feature store connection error."""
    pass


class FeatureSchemaMismatchError(FeatureServiceError):
    """Feature schema mismatch."""
    pass


class FeatureService:
    """
    Feature service for retrieving features from feature store.
    
    Priority:
    - Primary: Hopsworks (when available)
    - Fallback: Local store (only when explicitly enabled)
    """
    
    def __init__(
        self,
        primary_store: Optional[FeatureStoreInterface] = None,
        fallback_store: Optional[FeatureStoreInterface] = None,
        fallback_enabled: bool = False,
    ):
        """
        Initialize feature service.
        
        Args:
            primary_store: Primary feature store (Hopsworks)
            fallback_store: Fallback feature store (Local)
            fallback_enabled: Whether fallback is enabled
        """
        self.primary_store = primary_store
        self.fallback_store = fallback_store
        self.fallback_enabled = fallback_enabled
    
    def get_features(
        self,
        city: str,
        feature_names: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Get features for a city.
        
        Args:
            city: City name
            feature_names: Specific features to retrieve
            
        Returns:
            Dictionary of feature values
            
        Raises:
            FeatureStoreConnectionError: If both stores fail
            FeatureSchemaMismatchError: If schema doesn't match
        """
        # Try primary store first
        if self.primary_store:
            try:
                features = self._fetch_from_store(
                    self.primary_store, city, feature_names
                )
                return features
            except Exception as e:
                logger.warning(f"Primary store failed: {e}")
                
                # Try fallback if enabled
                if self.fallback_enabled and self.fallback_store:
                    logger.info("Falling back to local store")
                    try:
                        features = self._fetch_from_store(
                            self.fallback_store, city, feature_names
                        )
                        return features
                    except Exception as fallback_error:
                        logger.error(f"Fallback store also failed: {fallback_error}")
                        raise FeatureStoreConnectionError(
                            f"Both primary and fallback stores failed. "
                            f"Primary: {e}, Fallback: {fallback_error}"
                        )
                else:
                    raise FeatureStoreConnectionError(
                        f"Primary store failed: {e}"
                    )
        
        # No primary store configured
        raise FeatureStoreConnectionError("No feature store configured")
    
    def _fetch_from_store(
        self,
        store: FeatureStoreInterface,
        city: str,
        feature_names: Optional[list],
    ) -> Dict[str, Any]:
        """Fetch features from a specific store."""
        # Get latest features for city
        features_df = store.get_features(
            location_id=city,
            limit=1,
        )
        
        if features_df.empty:
            raise FeatureStoreConnectionError(
                f"No features found for city: {city}"
            )
        
        # Convert to dictionary
        features_dict = features_df.iloc[0].to_dict()
        
        # Filter by feature names if specified
        if feature_names:
            features_dict = {
                k: v for k, v in features_dict.items()
                if k in feature_names
            }
        
        return features_dict
    
    def validate_feature_schema(
        self,
        features: Dict[str, Any],
        required_features: list,
    ) -> bool:
        """
        Validate feature schema matches requirements.
        
        Args:
            features: Retrieved features
            required_features: Required feature names
            
        Returns:
            True if valid
            
        Raises:
            FeatureSchemaMismatchError: If schema doesn't match
        """
        missing = set(required_features) - set(features.keys())
        if missing:
            raise FeatureSchemaMismatchError(
                f"Missing required features: {missing}"
            )
        return True
    
    def is_connected(self) -> bool:
        """Check if any feature store is connected."""
        return self.primary_store is not None or (
            self.fallback_enabled and self.fallback_store is not None
        )


# Global feature service instance
_feature_service: Optional[FeatureService] = None


def get_feature_service() -> FeatureService:
    """Get global feature service instance."""
    global _feature_service
    if _feature_service is None:
        _feature_service = FeatureService()
    return _feature_service


def init_feature_service(
    primary_store: Optional[FeatureStoreInterface] = None,
    fallback_store: Optional[FeatureStoreInterface] = None,
    fallback_enabled: bool = False,
) -> FeatureService:
    """Initialize global feature service."""
    global _feature_service
    _feature_service = FeatureService(
        primary_store=primary_store,
        fallback_store=fallback_store,
        fallback_enabled=fallback_enabled,
    )
    return _feature_service
