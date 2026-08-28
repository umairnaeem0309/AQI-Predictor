"""
Tests for feature engineering pipeline.

Tests cover:
- Time features (hour, day, month, season, cyclical encoding)
- Lag features (all windows, edge cases)
- Rolling features (mean, std, min, max)
- Derived features (ratios, change rates, interactions)
- Missing value propagation
- Outlier capping
- Feature metadata
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import (
    FEATURE_VERSION,
    US_EPA_AQI_MAX,
    add_derived_features,
    add_lag_features,
    add_rolling_features,
    add_time_features,
    cap_outliers,
    engineer_features,
    get_feature_metadata,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def hourly_observation_data():
    """96 hours (4 days) of hourly observations for one city."""
    base_time = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
    timestamps = [base_time + timedelta(hours=i) for i in range(96)]

    np.random.seed(42)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "location_id": ["karachi"] * 96,
            "city_name": ["Karachi"] * 96,
            "temperature": 30 + np.random.randn(96) * 3,
            "humidity": 60 + np.random.randn(96) * 10,
            "wind_speed": 3 + np.random.rand(96) * 4,
            "pressure": 1010 + np.random.randn(96) * 5,
            "aqi": 100 + np.cumsum(np.random.randn(96) * 5),
            "pm25": 40 + np.random.rand(96) * 30,
            "pm10": 60 + np.random.rand(96) * 40,
            "co": 200 + np.random.rand(96) * 100,
            "no2": 20 + np.random.rand(96) * 20,
            "so2": 10 + np.random.rand(96) * 10,
            "o3": 40 + np.random.rand(96) * 30,
            "weather_condition": ["clear"] * 96,
            "data_source": ["openweather"] * 96,
        }
    )


@pytest.fixture
def multi_city_data():
    """Multi-city dataset for testing groupby behavior."""
    base_time = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
    timestamps = [base_time + timedelta(hours=i) for i in range(48)]

    rows = []
    for city in ["karachi", "lahore"]:
        for ts in timestamps:
            rows.append(
                {
                    "timestamp": ts,
                    "location_id": city,
                    "city_name": city.title(),
                    "temperature": 30 + np.random.randn() * 3,
                    "humidity": 60 + np.random.randn() * 10,
                    "aqi": 100 + np.random.randn() * 20,
                    "pm25": 40 + np.random.rand() * 30,
                    "data_source": "openweather",
                }
            )
    return pd.DataFrame(rows)


# =============================================================================
# Test Time Features
# =============================================================================


class TestTimeFeatures:
    """Tests for time-based feature extraction."""

    def test_hour_extraction(self, hourly_observation_data):
        """Hour is correctly extracted from timestamp."""
        df = add_time_features(hourly_observation_data)
        assert "hour" in df.columns
        assert df["hour"].iloc[0] == 0
        assert df["hour"].iloc[1] == 1
        assert df["hour"].iloc[23] == 23
        assert df["hour"].iloc[24] == 0  # Next day

    def test_day_of_week(self, hourly_observation_data):
        """Day of week is correctly extracted."""
        df = add_time_features(hourly_observation_data)
        assert "day_of_week" in df.columns
        # 2026-08-01 is Saturday (day 5)
        assert df["day_of_week"].iloc[0] == 5

    def test_month_extraction(self, hourly_observation_data):
        """Month is correctly extracted."""
        df = add_time_features(hourly_observation_data)
        assert "month" in df.columns
        assert (df["month"] == 8).all()  # All August

    def test_weekend_flag(self, hourly_observation_data):
        """Weekend flag is correctly computed."""
        df = add_time_features(hourly_observation_data)
        assert "is_weekend" in df.columns
        # Aug 1 (Sat=5) → is_weekend=1, Aug 2 (Sun=6) → is_weekend=1
        # Aug 3 (Mon=0) → is_weekend=0
        assert df["is_weekend"].iloc[0] == 1  # Saturday
        assert df["is_weekend"].iloc[24] == 1  # Sunday (index 24 = Aug 2 00:00)
        assert df["is_weekend"].iloc[48] == 0  # Monday (index 48 = Aug 3 00:00)

    def test_season_mapping(self, hourly_observation_data):
        """Season is correctly mapped from month."""
        df = add_time_features(hourly_observation_data)
        assert "season" in df.columns
        # August = summer = 2
        assert (df["season"] == 2).all()

    def test_cyclical_encoding(self, hourly_observation_data):
        """Cyclical encoding produces valid sin/cos values."""
        df = add_time_features(hourly_observation_data)
        assert "hour_sin" in df.columns
        assert "hour_cos" in df.columns
        # sin and cos should be in [-1, 1]
        assert df["hour_sin"].between(-1, 1).all()
        assert df["hour_cos"].between(-1, 1).all()

    def test_cyclical_continuity(self, hourly_observation_data):
        """Cyclical encoding avoids discontinuity at hour 23→0."""
        df = add_time_features(hourly_observation_data)
        # Hour 23 and hour 0 should have close sin/cos values
        # (they're adjacent on the circle)
        sin_23 = df["hour_sin"].iloc[23]
        sin_0 = df["hour_sin"].iloc[24]
        # sin(23π/12) ≈ sin(π/12) ≈ 0.259
        assert abs(sin_23 - sin_0) < 0.6


# =============================================================================
# Test Lag Features
# =============================================================================


class TestLagFeatures:
    """Tests for lag feature computation."""

    def test_lag_features_created(self, hourly_observation_data):
        """Lag features are created for all specified columns."""
        df = add_lag_features(hourly_observation_data)
        assert "aqi_lag_1h" in df.columns
        assert "aqi_lag_24h" in df.columns
        assert "pm25_lag_1h" in df.columns

    def test_lag_1h_correctness(self, hourly_observation_data):
        """Lag-1h value at position i equals base value at position i-1."""
        df = add_lag_features(hourly_observation_data, columns=["aqi"], lag_hours=[1])
        # First record should be NaN (no history)
        assert pd.isna(df["aqi_lag_1h"].iloc[0])
        # Second record should equal first record's AQI
        assert df["aqi_lag_1h"].iloc[1] == df["aqi"].iloc[0]

    def test_lag_24h_correctness(self, hourly_observation_data):
        """Lag-24h value at position i equals base value at position i-24."""
        df = add_lag_features(hourly_observation_data, columns=["aqi"], lag_hours=[24])
        # First 24 records should be NaN
        assert df["aqi_lag_24h"].iloc[:24].isna().all()
        # Record at index 24 should equal record at index 0
        assert df["aqi_lag_24h"].iloc[24] == df["aqi"].iloc[0]

    def test_lag_insufficient_history(self, hourly_observation_data):
        """Lag features have NaN for records with insufficient history."""
        df = add_lag_features(hourly_observation_data, columns=["aqi"], lag_hours=[72])
        # First 72 records should be NaN
        assert df["aqi_lag_72h"].iloc[:72].isna().all()

    def test_multi_city_lag_independence(self, multi_city_data):
        """Lag features are computed independently per city."""
        df = add_lag_features(multi_city_data, columns=["aqi"], lag_hours=[1])
        # Each city should have independent lag values
        karachi = df[df["location_id"] == "karachi"].sort_values("timestamp")
        lahore = df[df["location_id"] == "lahore"].sort_values("timestamp")
        # Lag values should differ between cities
        assert karachi["aqi_lag_1h"].iloc[1] != lahore["aqi_lag_1h"].iloc[1]


# =============================================================================
# Test Rolling Features
# =============================================================================


class TestRollingFeatures:
    """Tests for rolling window feature computation."""

    def test_rolling_features_created(self, hourly_observation_data):
        """Rolling features are created."""
        df = add_rolling_features(hourly_observation_data)
        assert "aqi_rolling_mean_24h" in df.columns
        assert "aqi_rolling_std_24h" in df.columns

    def test_rolling_mean_correctness(self, hourly_observation_data):
        """Rolling mean matches manual calculation."""
        df = add_rolling_features(
            hourly_observation_data,
            columns=["aqi"],
            windows={"aqi": [("6h", "mean")]},
        )
        # closed='left' excludes current period: at index 5, the 6h window
        # contains only indices 0-4 (5 values). At index 6, it contains 0-5 (6 values).
        expected_mean = hourly_observation_data["aqi"].iloc[:6].mean()
        assert abs(df["aqi_rolling_mean_6h"].iloc[6] - expected_mean) < 1e-10

    def test_rolling_std_non_negative(self, hourly_observation_data):
        """Rolling std is always non-negative."""
        df = add_rolling_features(hourly_observation_data)
        assert (df["aqi_rolling_std_24h"].dropna() >= 0).all()

    def test_rolling_min_max_order(self, hourly_observation_data):
        """Rolling min is always ≤ rolling max."""
        df = add_rolling_features(hourly_observation_data)
        valid = df["aqi_rolling_min_24h"].dropna() <= df["aqi_rolling_max_24h"].dropna()
        assert valid.all()

    def test_rolling_with_partial_window(self, hourly_observation_data):
        """Rolling works with partial windows (min_periods=1)."""
        df = add_rolling_features(
            hourly_observation_data,
            columns=["aqi"],
            windows={"aqi": [("24h", "mean")]},
        )
        # closed='left' excludes current period, so index 0 has no data to its left.
        # Index 1 has only index 0 → valid (min_periods=1).
        assert pd.isna(df["aqi_rolling_mean_24h"].iloc[0])  # No history left of index 0
        assert not pd.isna(df["aqi_rolling_mean_24h"].iloc[1])  # Has 1 hour of history


# =============================================================================
# Test Derived Features
# =============================================================================


class TestDerivedFeatures:
    """Tests for derived feature computation."""

    def test_aqi_change_rate_1h(self, hourly_observation_data):
        """AQI change rate 1h equals current minus lag."""
        df = add_lag_features(hourly_observation_data, columns=["aqi"], lag_hours=[1])
        df = add_derived_features(df)
        expected = df["aqi"] - df["aqi_lag_1h"]
        # Compare non-NaN values
        mask = df["aqi_change_rate_1h"].notna() & expected.notna()
        assert (df.loc[mask, "aqi_change_rate_1h"] == expected[mask]).all()

    def test_pm25_pm10_ratio(self, hourly_observation_data):
        """PM2.5/PM10 ratio is correctly computed."""
        df = add_derived_features(hourly_observation_data)
        expected = hourly_observation_data["pm25"] / hourly_observation_data["pm10"]
        mask = df["pm25_pm10_ratio"].notna()
        assert abs(df.loc[mask, "pm25_pm10_ratio"].iloc[0] - expected.iloc[0]) < 1e-10

    def test_ratio_zero_division(self):
        """Ratio features handle zero division gracefully."""
        df = pd.DataFrame(
            {
                "pm25": [50.0, 30.0],
                "pm10": [0.0, 60.0],
                "no2": [20.0, 0.0],
                "so2": [10.0, 10.0],
                "o3": [40.0, 40.0],
            }
        )
        df = add_derived_features(df)
        assert pd.isna(df["pm25_pm10_ratio"].iloc[0])  # pm10=0 → division by zero → NaN
        assert df["no2_so2_ratio"].iloc[1] == 0.0  # no2=0, so2=10 → 0/10 = 0.0, NOT NaN

    def test_temp_humidity_interaction(self, hourly_observation_data):
        """Temperature-humidity interaction is correctly computed."""
        df = add_derived_features(hourly_observation_data)
        expected = (
            hourly_observation_data["temperature"] * hourly_observation_data["humidity"] / 100
        )
        mask = df["temp_humidity_interaction"].notna()
        assert abs(df.loc[mask, "temp_humidity_interaction"].iloc[0] - expected.iloc[0]) < 1e-10


# =============================================================================
# Test Outlier Capping
# =============================================================================


class TestOutlierCapping:
    """Tests for outlier capping."""

    def test_aqi_capped_at_500(self):
        """AQI values above 500 are capped in feature columns."""
        df = pd.DataFrame(
            {
                "aqi": [600, 400, 300],
                "aqi_lag_1h": [550, 350, 250],
            }
        )
        df = cap_outliers(df)
        assert df["aqi"].iloc[0] == 500
        assert df["aqi_lag_1h"].iloc[0] == 500
        assert df["aqi"].iloc[1] == 400  # Below limit, unchanged

    def test_humidity_out_of_bounds(self):
        """Humidity values outside [0, 100] are set to NaN."""
        df = pd.DataFrame(
            {
                "humidity": [50, -10, 110, 75],
                "humidity_lag_1h": [60, -5, 105, 80],
            }
        )
        df = cap_outliers(df)
        assert df["humidity"].iloc[0] == 50
        assert pd.isna(df["humidity"].iloc[1])
        assert pd.isna(df["humidity"].iloc[2])
        assert df["humidity"].iloc[3] == 75


# =============================================================================
# Test Full Pipeline
# =============================================================================


class TestFullPipeline:
    """Tests for the complete feature engineering pipeline."""

    def test_pipeline_produces_features(self, hourly_observation_data):
        """Full pipeline produces all feature categories."""
        df = engineer_features(hourly_observation_data)
        # Time features
        assert "hour" in df.columns
        assert "season" in df.columns
        # Lag features
        assert "aqi_lag_1h" in df.columns
        # Rolling features
        assert "aqi_rolling_mean_24h" in df.columns
        # Derived features
        assert "aqi_change_rate_1h" in df.columns

    def test_pipeline_preserves_original_columns(self, hourly_observation_data):
        """Pipeline preserves original observation columns."""
        df = engineer_features(hourly_observation_data)
        assert "timestamp" in df.columns
        assert "location_id" in df.columns
        assert "aqi" in df.columns

    def test_pipeline_sorts_by_time(self, hourly_observation_data):
        """Pipeline sorts data by location and timestamp."""
        # Shuffle the data
        shuffled = hourly_observation_data.sample(frac=1, random_state=42)
        df = engineer_features(shuffled)
        # Check ordering
        assert df["timestamp"].is_monotonic_increasing

    def test_pipeline_preserves_missing_values(self, hourly_observation_data):
        """Pipeline preserves NaN values (no imputation)."""
        df = hourly_observation_data.copy()
        df.loc[0, "aqi"] = np.nan
        result = engineer_features(df)
        # Original NaN should be preserved
        assert pd.isna(result.loc[0, "aqi"])
        # Lag of NaN should also be NaN
        assert pd.isna(result.loc[1, "aqi_lag_1h"])

    def test_pipeline_metadata(self, hourly_observation_data):
        """Pipeline sets feature metadata."""
        df = engineer_features(hourly_observation_data)
        assert df.attrs.get("feature_version") == FEATURE_VERSION
        assert df.attrs.get("schema_version") == "1.0"
        assert "generation_timestamp" in df.attrs

    def test_empty_dataframe(self):
        """Pipeline handles empty DataFrame gracefully."""
        df = pd.DataFrame(columns=["timestamp", "location_id", "aqi"])
        result = engineer_features(df)
        assert result.empty

    def test_single_row(self):
        """Pipeline handles single-row DataFrame."""
        df = pd.DataFrame(
            {
                "timestamp": [datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)],
                "location_id": ["karachi"],
                "city_name": ["Karachi"],
                "aqi": [100],
                "temperature": [30.0],
                "humidity": [60.0],
                "pm25": [40.0],
                "data_source": ["openweather"],
            }
        )
        result = engineer_features(df)
        assert len(result) == 1
        assert "hour" in result.columns


# =============================================================================
# Test Feature Metadata
# =============================================================================


class TestFeatureMetadata:
    """Tests for feature metadata extraction."""

    def test_metadata_extraction(self, hourly_observation_data):
        """Metadata is correctly extracted from DataFrame."""
        df = engineer_features(hourly_observation_data)
        metadata = get_feature_metadata(df)
        assert metadata["feature_version"] == FEATURE_VERSION
        assert metadata["schema_version"] == "1.0"
        assert metadata["source_row_count"] == 96
        assert metadata["feature_count"] > 0
