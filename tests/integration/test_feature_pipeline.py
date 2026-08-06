"""
Integration test for feature engineering pipeline.

Tests the complete flow: raw observations → feature engineering → validation.
Uses mock data (API-shaped) to verify pipeline correctness without
requiring real API credentials.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone

from src.features.feature_engineering import (
    engineer_features,
    get_feature_metadata,
    FEATURE_VERSION,
)
from src.features.feature_validation import (
    full_feature_validation,
    get_feature_availability,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_raw_observations():
    """Simulated raw observations as would come from API manager.

    This mimics the output of APIManager.fetch_all_cities() — a DataFrame
    with StandardObservation columns.
    """
    base_time = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)

    rows = []
    for city_id, city_name, temp_base, aqi_base in [
        ("karachi", "Karachi", 32, 120),
        ("lahore", "Lahore", 36, 180),
        ("islamabad", "Islamabad", 30, 80),
    ]:
        for i in range(168):  # 7 days of hourly data
            ts = base_time + timedelta(hours=i)
            rows.append({
                "timestamp": ts,
                "location_id": city_id,
                "city_name": city_name,
                "temperature": temp_base + np.sin(i / 24 * 2 * np.pi) * 5 + np.random.randn() * 2,
                "humidity": 60 + np.cos(i / 24 * 2 * np.pi) * 15 + np.random.randn() * 5,
                "wind_speed": 3 + np.random.rand() * 4,
                "pressure": 1010 + np.random.randn() * 5,
                "aqi": aqi_base + np.sin(i / 48 * 2 * np.pi) * 30 + np.random.randn() * 10,
                "pm25": 40 + np.random.rand() * 30,
                "pm10": 60 + np.random.rand() * 40,
                "co": 200 + np.random.rand() * 100,
                "no2": 20 + np.random.rand() * 20,
                "so2": 10 + np.random.rand() * 10,
                "o3": 40 + np.random.rand() * 30,
                "weather_condition": ["clear"] * 1,
                "data_source": ["openweather"] * 1,
            })

    return pd.DataFrame(rows)


# =============================================================================
# Integration Tests
# =============================================================================


class TestFeaturePipelineEndToEnd:
    """End-to-end feature pipeline tests."""

    def test_pipeline_completes_for_all_cities(self, mock_raw_observations):
        """Feature engineering completes for all cities."""
        result = engineer_features(mock_raw_observations)

        # All cities should have features
        assert result["location_id"].nunique() == 3
        assert len(result) == 168 * 3  # 168 hours * 3 cities

    def test_features_created_for_all_categories(self, mock_raw_observations):
        """All feature categories are created."""
        result = engineer_features(mock_raw_observations)

        # Time features
        assert "hour" in result.columns
        assert "season" in result.columns
        assert "hour_sin" in result.columns

        # Lag features
        assert "aqi_lag_1h" in result.columns
        assert "aqi_lag_24h" in result.columns

        # Rolling features
        assert "aqi_rolling_mean_24h" in result.columns

        # Derived features
        assert "aqi_change_rate_1h" in result.columns
        assert "pm25_pm10_ratio" in result.columns

    def test_features_independent_per_city(self, mock_raw_observations):
        """Feature engineering is independent per city."""
        result = engineer_features(mock_raw_observations)

        # Check that lag features don't leak between cities
        for city_id in ["karachi", "lahore", "islamabad"]:
            city_data = result[result["location_id"] == city_id]
            # First record should have NaN lag
            assert pd.isna(city_data["aqi_lag_1h"].iloc[0])

    def test_validation_passes(self, mock_raw_observations):
        """Full validation passes on feature-engineered data."""
        result = engineer_features(mock_raw_observations)
        validation = full_feature_validation(result)

        assert len(validation["leakage_errors"]) == 0
        assert validation["total_features"] > 20
        assert validation["total_rows"] == 168 * 3

    def test_feature_metadata_set(self, mock_raw_observations):
        """Feature metadata is correctly set."""
        result = engineer_features(mock_raw_observations)
        metadata = get_feature_metadata(result)

        assert metadata["feature_version"] == FEATURE_VERSION
        assert metadata["source_row_count"] == 168 * 3
        assert metadata["feature_count"] > 20

    def test_feature_availability_documented(self, mock_raw_observations):
        """All features have documented availability."""
        availability = get_feature_availability()
        result = engineer_features(mock_raw_observations)

        # Check that all feature columns have availability docs
        feature_cols = [
            c for c in result.columns
            if c not in ["timestamp", "location_id", "city_name", "data_source", "weather_condition", "raw_response_time"]
        ]
        for col in feature_cols:
            assert col in availability, f"Feature {col} missing from availability documentation"

    def test_missing_values_preserved(self, mock_raw_observations):
        """Missing values are preserved, not imputed."""
        df = mock_raw_observations.copy()
        # Introduce some NaN values
        df.loc[0, "aqi"] = np.nan
        df.loc[5, "pm25"] = np.nan

        result = engineer_features(df)

        # Original NaN should be preserved
        assert pd.isna(result.loc[0, "aqi"])
        # Derived features from NaN input should also be NaN
        assert pd.isna(result.loc[0, "aqi_lag_1h"])

    def test_pipeline_with_minimal_data(self):
        """Pipeline works with minimal data (single city, few hours)."""
        base_time = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
        df = pd.DataFrame({
            "timestamp": [base_time + timedelta(hours=i) for i in range(10)],
            "location_id": ["karachi"] * 10,
            "city_name": ["Karachi"] * 10,
            "aqi": range(100, 110),
            "temperature": [30.0] * 10,
            "humidity": [60.0] * 10,
            "pm25": [40.0] * 10,
            "data_source": ["openweather"] * 10,
        })

        result = engineer_features(df)
        assert len(result) == 10
        assert "hour" in result.columns
        # Lag-72h should be all NaN (only 10 hours of data)
        if "aqi_lag_72h" in result.columns:
            assert result["aqi_lag_72h"].isna().all()
