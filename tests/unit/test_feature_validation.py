"""
Tests for feature validation and data leakage detection.

Tests cover:
- No future data leakage check
- Lag feature correctness validation
- Rolling feature plausibility validation
- Feature availability documentation
- Full validation pipeline
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone

from src.features.feature_validation import (
    check_no_future_leakage,
    validate_lag_features,
    validate_rolling_features,
    get_feature_availability,
    full_feature_validation,
)
from src.features.feature_engineering import (
    add_lag_features,
    add_rolling_features,
    add_time_features,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def clean_hourly_data():
    """Clean hourly data with no leakage."""
    base_time = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
    timestamps = [base_time + timedelta(hours=i) for i in range(96)]

    np.random.seed(42)
    aqi_values = 100 + np.cumsum(np.random.randn(96) * 5)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "location_id": ["karachi"] * 96,
        "aqi": aqi_values,
        "pm25": 40 + np.random.rand(96) * 30,
        "temperature": 30 + np.random.randn(96) * 3,
        "humidity": 60 + np.random.randn(96) * 10,
    })

    df = add_lag_features(df, columns=["aqi"], lag_hours=[1, 24])
    df = add_rolling_features(df, columns=["aqi"], windows={"aqi": [("24h", "mean")]})
    return df


@pytest.fixture
def data_with_leakage():
    """Data with intentional leakage (future data in lag feature)."""
    base_time = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
    timestamps = [base_time + timedelta(hours=i) for i in range(10)]

    df = pd.DataFrame({
        "timestamp": timestamps,
        "location_id": ["karachi"] * 10,
        "aqi": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
    })

    # Create a lag feature with incorrect values (simulating leakage)
    # Position 0 should have NaN for lag_1h, but we set it to a future value
    df["aqi_lag_1h"] = [110, 100, 110, 120, 130, 140, 150, 160, 170, 180]
    return df


# =============================================================================
# Test Leakage Detection
# =============================================================================


class TestLeakageDetection:
    """Tests for data leakage detection."""

    def test_no_leakage_in_clean_data(self, clean_hourly_data):
        """Clean data passes leakage check."""
        errors = check_no_future_leakage(
            clean_hourly_data,
            feature_columns=["aqi_lag_1h", "aqi_lag_24h", "aqi_rolling_mean_24h"],
        )
        assert len(errors) == 0

    def test_detects_leakage(self, data_with_leakage):
        """Leakage is detected in tampered data."""
        errors = check_no_future_leakage(
            data_with_leakage,
            feature_columns=["aqi_lag_1h"],
        )
        # Should detect that lag_1h at position 0 doesn't match expected
        assert len(errors) > 0

    def test_empty_dataframe(self):
        """Empty DataFrame passes leakage check."""
        df = pd.DataFrame(columns=["timestamp", "location_id", "aqi"])
        errors = check_no_future_leakage(df, feature_columns=["aqi"])
        assert len(errors) == 0


# =============================================================================
# Test Lag Validation
# =============================================================================


class TestLagValidation:
    """Tests for lag feature correctness validation."""

    def test_valid_lag_features(self, clean_hourly_data):
        """Valid lag features pass validation."""
        results = validate_lag_features(
            clean_hourly_data,
            lag_columns=["aqi_lag_1h", "aqi_lag_24h"],
        )
        assert results["aqi_lag_1h"] is True
        assert results["aqi_lag_24h"] is True

    def test_invalid_lag_features(self, data_with_leakage):
        """Invalid lag features fail validation."""
        results = validate_lag_features(
            data_with_leakage,
            lag_columns=["aqi_lag_1h"],
        )
        assert results["aqi_lag_1h"] is False

    def test_nonexistent_column(self, clean_hourly_data):
        """Nonexistent column fails validation."""
        results = validate_lag_features(
            clean_hourly_data,
            lag_columns=["nonexistent_lag_1h"],
        )
        assert results["nonexistent_lag_1h"] is False


# =============================================================================
# Test Rolling Validation
# =============================================================================


class TestRollingValidation:
    """Tests for rolling feature plausibility validation."""

    def test_valid_rolling_features(self, clean_hourly_data):
        """Valid rolling features pass validation."""
        results = validate_rolling_features(
            clean_hourly_data,
            rolling_columns=["aqi_rolling_mean_24h"],
        )
        assert results["aqi_rolling_mean_24h"] is True

    def test_rolling_std_non_negative(self):
        """Rolling std is validated as non-negative."""
        df = pd.DataFrame({
            "location_id": ["k"] * 10,
            "timestamp": pd.date_range("2026-08-01", periods=10, freq="h", tz="UTC"),
            "aqi_rolling_std_24h": [1.0, 2.0, 3.0, -1.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        })
        results = validate_rolling_features(df, rolling_columns=["aqi_rolling_std_24h"])
        assert results["aqi_rolling_std_24h"] is False

    def test_min_greater_than_max(self):
        """Rolling min > max is detected."""
        df = pd.DataFrame({
            "location_id": ["k"] * 10,
            "timestamp": pd.date_range("2026-08-01", periods=10, freq="h", tz="UTC"),
            "aqi_rolling_min_24h": [100.0] * 10,
            "aqi_rolling_max_24h": [50.0] * 10,  # Min > Max!
        })
        results = validate_rolling_features(
            df, rolling_columns=["aqi_rolling_min_24h", "aqi_rolling_max_24h"]
        )
        assert results["aqi_rolling_min_24h"] is False


# =============================================================================
# Test Feature Availability
# =============================================================================


class TestFeatureAvailability:
    """Tests for feature availability documentation."""

    def test_time_features_available_immediately(self):
        """Time features are available at observation time."""
        availability = get_feature_availability()
        assert "immediately available" in availability["hour"]
        assert "immediately available" in availability["season"]

    def test_lag_features_available_at_t(self):
        """Lag features are available at prediction time t."""
        availability = get_feature_availability()
        assert "available at prediction time" in availability["aqi_lag_1h"]
        assert "available at prediction time" in availability["aqi_lag_72h"]

    def test_rolling_features_available_at_t(self):
        """Rolling features are available at prediction time t."""
        availability = get_feature_availability()
        assert "available at prediction time" in availability["aqi_rolling_mean_24h"]

    def test_derived_features_available_at_t(self):
        """Derived features are available at prediction time t."""
        availability = get_feature_availability()
        assert "available at prediction time" in availability["aqi_change_rate_1h"]

    def test_all_features_documented(self):
        """All expected features have availability documentation."""
        availability = get_feature_availability()
        expected_features = [
            "hour", "day_of_week", "month", "season",
            "aqi_lag_1h", "aqi_lag_24h", "aqi_lag_72h",
            "aqi_rolling_mean_24h",
            "aqi_change_rate_1h", "pm25_pm10_ratio",
            "aqi", "temperature", "humidity",
        ]
        for feat in expected_features:
            assert feat in availability, f"Feature {feat} missing from availability docs"


# =============================================================================
# Test Full Validation
# =============================================================================


class TestFullValidation:
    """Tests for full feature validation pipeline."""

    def test_full_validation_clean_data(self, clean_hourly_data):
        """Clean data passes full validation."""
        results = full_feature_validation(clean_hourly_data)
        assert len(results["leakage_errors"]) == 0
        assert results["total_features"] > 0
        assert results["total_rows"] == 96

    def test_full_validation_completeness(self, clean_hourly_data):
        """Completeness metrics are computed."""
        results = full_feature_validation(clean_hourly_data)
        assert "completeness" in results
        assert "aqi" in results["completeness"]
        assert results["completeness"]["aqi"]["total"] == 96

    def test_full_validation_with_custom_columns(self, clean_hourly_data):
        """Full validation with custom column list."""
        results = full_feature_validation(
            clean_hourly_data,
            feature_columns=["aqi", "aqi_lag_1h"],
        )
        assert results["total_features"] == 2
