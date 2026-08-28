"""
Tests for dataset builder — target generation, splitting, and metadata.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.data.dataset_builder import (
    add_source_quality_metadata,
    build_dataset,
    generate_dataset_version,
    generate_targets,
    split_chronological,
    validate_no_target_leakage,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def hourly_observations():
    """6 days of hourly observations for one city."""
    base_time = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
    timestamps = [base_time + timedelta(hours=i) for i in range(144)]

    np.random.seed(42)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "location_id": ["karachi"] * 144,
            "city_name": ["Karachi"] * 144,
            "temperature": 30 + np.random.randn(144) * 3,
            "humidity": 60 + np.random.randn(144) * 10,
            "wind_speed": 3 + np.random.rand(144) * 4,
            "pressure": 1010 + np.random.randn(144) * 5,
            "aqi": 100 + np.cumsum(np.random.randn(144) * 3),
            "pm25": 40 + np.random.rand(144) * 20,
            "pm10": 60 + np.random.rand(144) * 30,
            "co": 200 + np.random.rand(144) * 80,
            "no2": 20 + np.random.rand(144) * 15,
            "so2": 10 + np.random.rand(144) * 8,
            "o3": 40 + np.random.rand(144) * 20,
            "weather_condition": ["clear"] * 144,
            "data_source": ["openweather"] * 144,
        }
    )


# =============================================================================
# Test Source Quality Metadata
# =============================================================================


class TestSourceQualityMetadata:
    """Tests for source quality tracking."""

    def test_weather_available(self, hourly_observations):
        """Weather availability is correctly detected."""
        df = add_source_quality_metadata(hourly_observations)
        assert "weather_available" in df.columns
        assert (df["weather_available"] == 1).all()

    def test_aqi_available(self, hourly_observations):
        """AQI availability is correctly detected."""
        df = add_source_quality_metadata(hourly_observations)
        assert "aqi_available" in df.columns
        assert (df["aqi_available"] == 1).all()

    def test_missing_weather(self, hourly_observations):
        """Missing weather fields are detected."""
        df = hourly_observations.copy()
        df["temperature"] = np.nan
        df["humidity"] = np.nan
        df["wind_speed"] = np.nan
        df["pressure"] = np.nan
        result = add_source_quality_metadata(df)
        assert (result["weather_available"] == 0).all()

    def test_sources_used(self, hourly_observations):
        """Sources used field is populated."""
        df = add_source_quality_metadata(hourly_observations)
        assert "sources_used" in df.columns
        assert (df["sources_used"] == "openweather").all()


# =============================================================================
# Test Target Generation
# =============================================================================


class TestTargetGeneration:
    """Tests for target variable generation."""

    def test_targets_created(self, hourly_observations):
        """Target columns are created for all horizons."""
        df = generate_targets(hourly_observations)
        assert "target_aqi_24h" in df.columns
        assert "target_aqi_48h" in df.columns
        assert "target_aqi_72h" in df.columns

    def test_target_24h_correctness(self, hourly_observations):
        """Target 24h equals AQI value 24 hours ahead."""
        df = generate_targets(hourly_observations, horizons=[24])
        # Row 0's target should be row 24's AQI
        assert df["target_aqi_24h"].iloc[0] == hourly_observations["aqi"].iloc[24]
        # Row 1's target should be row 25's AQI
        assert df["target_aqi_24h"].iloc[1] == hourly_observations["aqi"].iloc[25]

    def test_target_72h_correctness(self, hourly_observations):
        """Target 72h equals AQI value 72 hours ahead."""
        df = generate_targets(hourly_observations, horizons=[72])
        assert df["target_aqi_72h"].iloc[0] == hourly_observations["aqi"].iloc[72]

    def test_target_nan_at_end(self, hourly_observations):
        """Last N rows have NaN targets (no future data)."""
        df = generate_targets(hourly_observations, horizons=[24, 48, 72])
        # Last 24 rows should have NaN for 24h target
        assert df["target_aqi_24h"].iloc[-24:].isna().all()
        # Last 48 rows should have NaN for 48h target
        assert df["target_aqi_48h"].iloc[-48:].isna().all()
        # Last 72 rows should have NaN for 72h target
        assert df["target_aqi_72h"].iloc[-72:].isna().all()

    def test_no_aqi_column(self, hourly_observations):
        """Missing AQI column produces warning and no targets."""
        df = hourly_observations.drop(columns=["aqi"])
        result = generate_targets(df)
        assert "target_aqi_24h" not in result.columns


# =============================================================================
# Test Target Leakage Validation
# =============================================================================


class TestTargetLeakageValidation:
    """Tests for target leakage detection."""

    def test_no_leakage_in_clean_data(self, hourly_observations):
        """Clean data passes leakage validation."""
        df = generate_targets(hourly_observations)
        errors = validate_no_target_leakage(
            df,
            feature_columns=["temperature", "humidity", "aqi_lag_1h"],
            target_columns=["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"],
        )
        assert len(errors) == 0

    def test_detects_target_in_features(self, hourly_observations):
        """Target column in feature columns is detected."""
        df = generate_targets(hourly_observations)
        errors = validate_no_target_leakage(
            df,
            feature_columns=["temperature", "target_aqi_24h"],
            target_columns=["target_aqi_24h"],
        )
        assert len(errors) > 0
        assert "target_aqi_24h" in errors[0]


# =============================================================================
# Test Chronological Split
# =============================================================================


class TestChronologicalSplit:
    """Tests for chronological train/val/test splitting."""

    def test_split_sizes(self, hourly_observations):
        """Split produces correct proportions."""
        train, val, test = split_chronological(hourly_observations)
        total = len(train) + len(val) + len(test)
        assert total == len(hourly_observations)
        # Training should be largest
        assert len(train) > len(val)
        assert len(train) > len(test)

    def test_chronological_ordering(self, hourly_observations):
        """Each split is chronologically ordered."""
        train, val, test = split_chronological(hourly_observations)
        assert train["timestamp"].is_monotonic_increasing
        assert val["timestamp"].is_monotonic_increasing
        assert test["timestamp"].is_monotonic_increasing

    def test_no_overlap(self, hourly_observations):
        """No overlapping timestamps between splits."""
        train, val, test = split_chronological(hourly_observations)
        train_times = set(train["timestamp"])
        val_times = set(val["timestamp"])
        test_times = set(test["timestamp"])
        assert len(train_times & val_times) == 0
        assert len(train_times & test_times) == 0
        assert len(val_times & test_times) == 0

    def test_continuous_timeline(self, hourly_observations):
        """Split creates continuous timeline (train ends where val starts)."""
        train, val, test = split_chronological(hourly_observations)
        if len(val) > 0 and len(train) > 0:
            train_end = train["timestamp"].max()
            val_start = val["timestamp"].min()
            # Val should start after or at train end
            assert val_start >= train_end

    def test_split_ratios(self, hourly_observations):
        """Custom split ratios are respected."""
        train, val, test = split_chronological(hourly_observations, train_ratio=0.7, val_ratio=0.2)
        total = len(hourly_observations)
        assert abs(len(train) / total - 0.7) < 0.05
        assert abs(len(val) / total - 0.2) < 0.05


# =============================================================================
# Test Dataset Version
# =============================================================================


class TestDatasetVersion:
    """Tests for dataset version generation."""

    def test_version_format(self):
        """Version string follows expected format: vYYYYMMDD_<6-char-hash>."""
        version = generate_dataset_version()
        assert version.startswith("v")
        # v + 8-digit date + _ + 6-char hex = 16 chars
        assert len(version) == 16  # v20260826_159eb2
        parts = version.split("_")
        assert len(parts) == 2
        assert parts[0][1:].isdigit()  # date part after v
        assert len(parts[1]) == 6  # hash suffix

    def test_unique_versions(self):
        """Versions from different timestamps are unique."""
        import time

        v1 = generate_dataset_version()
        time.sleep(1.1)  # Ensure different second for uniqueness
        v2 = generate_dataset_version()
        assert v1 != v2


# =============================================================================
# Test Full Build
# =============================================================================


class TestFullBuild:
    """Tests for complete dataset building pipeline."""

    def test_build_produces_splits(self, hourly_observations):
        """Build produces train/val/test splits."""
        result = build_dataset(hourly_observations, save=False)
        assert "train" in result
        assert "val" in result
        assert "test" in result
        assert "metadata" in result

    def test_build_metadata(self, hourly_observations):
        """Build generates correct metadata."""
        result = build_dataset(hourly_observations, save=False)
        meta = result["metadata"]
        assert "dataset_version" in meta
        assert "feature_version" in meta
        assert meta["total_records"] == len(hourly_observations)
        assert meta["feature_count"] > 0
        assert meta["target_count"] == 3

    def test_build_no_leakage(self, hourly_observations):
        """Build detects no leakage errors."""
        result = build_dataset(hourly_observations, save=False)
        assert len(result["metadata"]["leakage_errors"]) == 0

    def test_build_empty_dataframe(self):
        """Build handles empty DataFrame."""
        df = pd.DataFrame(columns=["timestamp", "location_id", "aqi"])
        result = build_dataset(df, save=False)
        assert result == {}
