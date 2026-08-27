"""
Live Feature Adapter — Bridges FeatureService and LocalStore for live predictions.

For production predictions, the API needs to:
1. Get current weather/pollution data for a city
2. Engineer features on-the-fly
3. Return features in the format the model expects

This adapter loads the latest features from processed data and
provides them in the format FeatureService expects.
"""

import logging
from typing import Dict, Any, Optional

import pandas as pd

from src.feature_store.base import FeatureStoreInterface

logger = logging.getLogger(__name__)


class LiveFeatureAdapter:
    """
    Adapter that provides live features for predictions.

    Loads the latest features from processed data files and returns
    them in the format expected by the prediction pipeline.
    """

    def __init__(self, features_path: str = "data/processed/test_features.csv",
                 metadata_path: str = "models/production/model_metadata.json"):
        """
        Initialize the adapter.

        Args:
            features_path: Path to processed features CSV.
            metadata_path: Path to model metadata JSON (for feature list).
        """
        self.features_path = features_path
        self.metadata_path = metadata_path
        self._features_df = None
        self._model_features = None

    def _load_features(self):
        """Load features from CSV if not already loaded."""
        if self._features_df is None:
            try:
                self._features_df = pd.read_csv(self.features_path)
                logger.info("Loaded %d feature rows from %s", len(self._features_df), self.features_path)
            except Exception as e:
                logger.error("Failed to load features: %s", e)
                raise

        # Load model feature list to match exactly
        if self._model_features is None:
            try:
                from pathlib import Path
                meta_path = Path(self.metadata_path)
                if meta_path.exists():
                    import json
                    with open(meta_path) as f:
                        meta = json.load(f)
                    self._model_features = meta.get("feature_columns", [])
                    logger.info("Loaded %d model features from metadata", len(self._model_features))
            except Exception as e:
                logger.warning("Could not load model metadata: %s", e)

    def get_latest_features(self, city: str) -> Dict[str, Any]:
        """
        Get the latest features for a city.

        Args:
            city: City name (karachi, lahore, islamabad).

        Returns:
            Dictionary of feature values.
        """
        self._load_features()

        # Filter by city
        city_data = self._features_df[self._features_df["location_id"] == city]

        if city_data.empty:
            raise ValueError(f"No features found for city: {city}")

        # Get the latest row
        latest = city_data.iloc[-1]

        # Use model's exact feature list to match training
        features = {}
        if self._model_features:
            for col in self._model_features:
                if col in latest.index:
                    val = latest[col]
                    features[col] = float(val) if pd.notna(val) else 0.0
        else:
            # Fallback: exclude non-feature columns
            exclude_cols = {"timestamp", "location_id", "city_name", "data_source",
                           "aqi_category", "aqi_standard", "aqi_method",
                           "aqi_method_version", "aqi_source"}
            for col in self._features_df.columns:
                if col not in exclude_cols:
                    val = latest[col]
                    if pd.notna(val):
                        features[col] = float(val) if isinstance(val, (int, float)) else val

        logger.info("Retrieved %d features for city=%s", len(features), city)
        return features


# Global adapter instance
_live_adapter: Optional[LiveFeatureAdapter] = None


def get_live_adapter() -> LiveFeatureAdapter:
    """Get or create the global live feature adapter."""
    global _live_adapter
    if _live_adapter is None:
        _live_adapter = LiveFeatureAdapter()
    return _live_adapter
