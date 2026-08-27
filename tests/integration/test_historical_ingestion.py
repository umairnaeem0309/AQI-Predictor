"""
Integration Tests for Historical Data Ingestion Pipeline.

Tests:
- Weather + Air Quality merge
- AQI target calculation
- Data validation
- End-to-end pipeline with mocked API responses
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.data.historical_ingestion import (
    merge_weather_and_air_quality,
    calculate_aqi_targets,
    validate_dataset,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_weather_df():
    """Sample weather DataFrame matching Open-Meteo output format."""
    n_hours = 100
    timestamps = pd.date_range(
        start="2023-06-01", periods=n_hours, freq="h", tz="UTC"
    )
    np.random.seed(42)
    return pd.DataFrame({
        "timestamp": timestamps,
        "location_id": "karachi",
        "city_name": "Karachi",
        "temperature": 30 + np.random.randn(n_hours) * 3,
        "humidity": 65 + np.random.randn(n_hours) * 10,
        "pressure": 1012 + np.random.randn(n_hours) * 2,
        "wind_speed": 3 + np.random.rand(n_hours) * 5,
        "wind_direction": np.random.uniform(0, 360, n_hours),
        "cloud_cover": np.clip(50 + np.random.randn(n_hours) * 20, 0, 100),
        "precipitation": np.random.exponential(0.1, n_hours),
        "data_source": "open_meteo_weather",
    })


@pytest.fixture
def sample_aq_df():
    """Sample air quality DataFrame matching Open-Meteo output format."""
    n_hours = 100
    timestamps = pd.date_range(
        start="2023-06-01", periods=n_hours, freq="h", tz="UTC"
    )
    np.random.seed(42)
    return pd.DataFrame({
        "timestamp": timestamps,
        "location_id": "karachi",
        "pm25": 25 + np.random.rand(n_hours) * 30,
        "pm10": 40 + np.random.rand(n_hours) * 40,
        "co": 400 + np.random.rand(n_hours) * 200,
        "no2": 25 + np.random.rand(n_hours) * 20,
        "so2": 10 + np.random.rand(n_hours) * 10,
        "o3": 40 + np.random.rand(n_hours) * 30,
        "us_aqi_open_meteo": 70 + np.random.randint(0, 40, n_hours),
        "data_source": "open_meteo_air_quality",
    })


@pytest.fixture
def sample_multi_city_weather():
    """Weather data for multiple cities."""
    n_hours = 48
    timestamps = pd.date_range(
        start="2023-06-01", periods=n_hours, freq="h", tz="UTC"
    )
    frames = []
    for city_id, city_name, base_temp in [
        ("karachi", "Karachi", 30),
        ("lahore", "Lahore", 33),
        ("islamabad", "Islamabad", 28),
    ]:
        np.random.seed(hash(city_id) % 2**31)
        frames.append(pd.DataFrame({
            "timestamp": timestamps,
            "location_id": city_id,
            "city_name": city_name,
            "temperature": base_temp + np.random.randn(n_hours) * 3,
            "humidity": 60 + np.random.randn(n_hours) * 10,
            "pressure": 1012 + np.random.randn(n_hours) * 2,
            "wind_speed": 3 + np.random.rand(n_hours) * 4,
            "data_source": "open_meteo_weather",
        }))
    return pd.concat(frames, ignore_index=True)


# ============================================================================
# Merge Tests
# ============================================================================


class TestMergeWeatherAndAirQuality:
    """Tests for merging weather and air quality DataFrames."""

    def test_merge_basic(self, sample_weather_df, sample_aq_df):
        """Basic merge produces combined DataFrame."""
        merged = merge_weather_and_air_quality(sample_weather_df, sample_aq_df)

        assert not merged.empty
        assert "temperature" in merged.columns
        assert "pm25" in merged.columns
        assert "humidity" in merged.columns
        assert "pm10" in merged.columns

    def test_merge_row_count(self, sample_weather_df, sample_aq_df):
        """Merge preserves row count when timestamps align."""
        merged = merge_weather_and_air_quality(sample_weather_df, sample_aq_df)
        # Should have same number of rows as input (timestamps match exactly)
        assert len(merged) == len(sample_weather_df)

    def test_merge_preserves_timestamps(self, sample_weather_df, sample_aq_df):
        """Merged timestamps are preserved correctly."""
        merged = merge_weather_and_air_quality(sample_weather_df, sample_aq_df)
        assert pd.api.types.is_datetime64_any_dtype(merged["timestamp"])

    def test_merge_preserves_location(self, sample_weather_df, sample_aq_df):
        """Location IDs are preserved after merge."""
        merged = merge_weather_and_air_quality(sample_weather_df, sample_aq_df)
        assert (merged["location_id"] == "karachi").all()

    def test_merge_empty_weather(self, sample_aq_df):
        """Merge with empty weather returns air quality only."""
        merged = merge_weather_and_air_quality(pd.DataFrame(), sample_aq_df)
        assert not merged.empty
        assert "pm25" in merged.columns

    def test_merge_empty_aq(self, sample_weather_df):
        """Merge with empty air quality returns weather only."""
        merged = merge_weather_and_air_quality(sample_weather_df, pd.DataFrame())
        assert not merged.empty
        assert "temperature" in merged.columns

    def test_merge_both_empty(self):
        """Merge with both empty returns empty DataFrame."""
        merged = merge_weather_and_air_quality(pd.DataFrame(), pd.DataFrame())
        assert merged.empty

    def test_merge_outer_join_fills_missing(self):
        """Outer join fills missing values with NaN."""
        weather = pd.DataFrame({
            "timestamp": pd.to_datetime(["2023-01-01T00:00", "2023-01-01T01:00"], utc=True),
            "location_id": ["karachi", "karachi"],
            "temperature": [25.0, 26.0],
        })
        aq = pd.DataFrame({
            "timestamp": pd.to_datetime(["2023-01-01T01:00", "2023-01-01T02:00"], utc=True),
            "location_id": ["karachi", "karachi"],
            "pm25": [30.0, 35.0],
        })
        merged = merge_weather_and_air_quality(weather, aq)
        assert len(merged) == 3  # 3 unique timestamps
        # First row: weather only → pm25 should be NaN
        assert pd.isna(merged.iloc[0]["pm25"])
        # Last row: aq only → temperature should be NaN
        assert pd.isna(merged.iloc[2]["temperature"])


# ============================================================================
# AQI Calculation Tests
# ============================================================================


class TestCalculateAQITargets:
    """Tests for EPA AQI target calculation."""

    def test_aqi_calculation_adds_columns(self, sample_weather_df, sample_aq_df):
        """AQI calculation adds required columns."""
        merged = merge_weather_and_air_quality(sample_weather_df, sample_aq_df)
        result = calculate_aqi_targets(merged)

        assert "aqi" in result.columns
        assert "pm25_aqi" in result.columns
        assert "pm10_aqi" in result.columns
        assert "aqi_dominant_pollutant" in result.columns
        assert "aqi_category" in result.columns
        assert "aqi_standard" in result.columns
        assert "aqi_derived" in result.columns

    def test_aqi_values_in_range(self, sample_weather_df, sample_aq_df):
        """All AQI values are in valid range (0-500)."""
        merged = merge_weather_and_air_quality(sample_weather_df, sample_aq_df)
        result = calculate_aqi_targets(merged)

        valid_aqi = result["aqi"].dropna()
        assert (valid_aqi >= 0).all()
        assert (valid_aqi <= 500).all()

    def test_aqi_dominant_pollutant(self, sample_weather_df, sample_aq_df):
        """Dominant pollutant is either pm25 or pm10."""
        merged = merge_weather_and_air_quality(sample_weather_df, sample_aq_df)
        result = calculate_aqi_targets(merged)

        valid_dominant = result["aqi_dominant_pollutant"].dropna()
        assert valid_dominant.isin(["pm25", "pm10"]).all()

    def test_aqi_invariant(self, sample_weather_df, sample_aq_df):
        """AQI equals max(pm25_aqi, pm10_aqi) when both valid."""
        merged = merge_weather_and_air_quality(sample_weather_df, sample_aq_df)
        result = calculate_aqi_targets(merged)

        both_valid = result["pm25_aqi"].notna() & result["pm10_aqi"].notna()
        if both_valid.any():
            subset = result[both_valid]
            expected_max = subset[["pm25_aqi", "pm10_aqi"]].max(axis=1)
            pd.testing.assert_series_equal(
                subset["aqi"].reset_index(drop=True),
                expected_max.reset_index(drop=True),
                check_names=False,
            )

    def test_aqi_high_pm25_selects_pm25(self):
        """When PM2.5 AQI > PM10 AQI, dominant is pm25."""
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2023-01-01T00:00"], utc=True),
            "location_id": ["karachi"],
            "pm25": [150.0],  # Very high PM2.5 → high AQI
            "pm10": [30.0],   # Low PM10 → low AQI
        })
        result = calculate_aqi_targets(df)
        assert result.iloc[0]["aqi_dominant_pollutant"] == "pm25"

    def test_aqi_high_pm10_selects_pm10(self):
        """When PM10 AQI > PM2.5 AQI, dominant is pm10."""
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2023-01-01T00:00"], utc=True),
            "location_id": ["karachi"],
            "pm25": [10.0],   # Low PM2.5
            "pm10": [200.0],  # High PM10 → high AQI
        })
        result = calculate_aqi_targets(df)
        assert result.iloc[0]["aqi_dominant_pollutant"] == "pm10"

    def test_aqi_missing_pollutants(self):
        """Missing pollutant values result in None AQI."""
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2023-01-01T00:00"], utc=True),
            "location_id": ["karachi"],
            "pm25": [None],
            "pm10": [None],
        })
        result = calculate_aqi_targets(df)
        assert result.iloc[0]["aqi"] is None
        assert result.iloc[0]["aqi_dominant_pollutant"] is None

    def test_aqi_metadata(self, sample_weather_df, sample_aq_df):
        """AQI metadata is correctly set."""
        merged = merge_weather_and_air_quality(sample_weather_df, sample_aq_df)
        result = calculate_aqi_targets(merged)

        assert (result["aqi_standard"] == "US_EPA").all()
        assert result.iloc[0]["aqi_derived"] == True
        assert "EPA" in result.iloc[0]["aqi_method_version"]


# ============================================================================
# Validation Tests
# ============================================================================


class TestValidateDataset:
    """Tests for data quality validation."""

    def test_validation_pass(self, sample_weather_df, sample_aq_df):
        """Clean dataset passes validation."""
        merged = merge_weather_and_air_quality(sample_weather_df, sample_aq_df)
        result = calculate_aqi_targets(merged)
        report = validate_dataset(result)

        assert report["status"] in ("PASS", "WARNING")
        assert report["total_rows"] == len(result)
        assert "karachi" in report["cities"]

    def test_validation_catches_negative_pollutants(self):
        """Negative pollutant values are flagged."""
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2023-01-01"], utc=True),
            "location_id": ["karachi"],
            "pm25": [-5.0],
            "pm10": [30.0],
            "aqi": [50],
        })
        report = validate_dataset(df)
        assert any("Negative" in issue for issue in report["quality_issues"])

    def test_validation_catches_extreme_temperature(self):
        """Extreme temperature values are flagged."""
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2023-01-01"], utc=True),
            "location_id": ["karachi"],
            "temperature": [999.0],
            "aqi": [50],
        })
        report = validate_dataset(df)
        assert any("Extreme temperature" in issue for issue in report["quality_issues"])

    def test_validation_catches_invalid_humidity(self):
        """Humidity outside 0-100 is flagged."""
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2023-01-01"], utc=True),
            "location_id": ["karachi"],
            "humidity": [150.0],
            "aqi": [50],
        })
        report = validate_dataset(df)
        assert any("humidity" in issue.lower() for issue in report["quality_issues"])

    def test_validation_catches_duplicates(self):
        """Duplicate (timestamp, location_id) pairs are flagged."""
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2023-01-01", "2023-01-01"], utc=True),
            "location_id": ["karachi", "karachi"],
            "aqi": [50, 60],
        })
        report = validate_dataset(df)
        assert any("Duplicate" in issue for issue in report["quality_issues"])

    def test_validation_catches_out_of_range_aqi(self):
        """AQI outside 0-500 is flagged."""
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2023-01-01"], utc=True),
            "location_id": ["karachi"],
            "aqi": [600],
        })
        report = validate_dataset(df)
        assert any("500" in issue for issue in report["quality_issues"])

    def test_validation_empty_dataset(self):
        """Empty dataset is handled gracefully."""
        report = validate_dataset(pd.DataFrame())
        assert report["status"] == "FAIL"

    def test_validation_date_range(self, sample_weather_df, sample_aq_df):
        """Validation reports correct date range."""
        merged = merge_weather_and_air_quality(sample_weather_df, sample_aq_df)
        report = validate_dataset(merged)
        assert "date_range" in report
        assert "start" in report["date_range"]
        assert "end" in report["date_range"]


# ============================================================================
# End-to-End Pipeline Test (Mocked APIs)
# ============================================================================


class TestIngestionPipeline:
    """End-to-end tests with mocked Open-Meteo API responses."""

    @patch("src.data.providers.open_meteo_weather.OpenMeteoWeatherProvider._fetch_chunk")
    @patch("src.data.providers.open_meteo_air_quality.OpenMeteoAirQualityProvider._fetch_chunk")
    def test_full_pipeline_mocked(self, mock_aq_fetch, mock_weather_fetch):
        """Full pipeline works with mocked API responses."""
        from src.data.historical_ingestion import run_historical_ingestion

        # Mock weather response
        n_hours = 48
        timestamps = pd.date_range(
            start="2023-06-01", periods=n_hours, freq="h", tz="UTC"
        ).strftime("%Y-%m-%dT%H:%M").tolist()

        mock_weather_fetch.return_value = {
            "hourly": {
                "time": timestamps,
                "temperature_2m": [28.0 + i * 0.1 for i in range(n_hours)],
                "relative_humidity_2m": [65.0] * n_hours,
                "surface_pressure": [1012.0] * n_hours,
                "wind_speed_10m": [3.0] * n_hours,
                "wind_direction_10m": [180.0] * n_hours,
                "cloud_cover": [40.0] * n_hours,
                "precipitation": [0.0] * n_hours,
            }
        }

        # Mock AQ response
        mock_aq_fetch.return_value = {
            "hourly": {
                "time": timestamps,
                "pm2_5": [25.0 + i * 0.5 for i in range(n_hours)],
                "pm10": [40.0 + i * 0.3 for i in range(n_hours)],
                "carbon_monoxide": [400.0] * n_hours,
                "nitrogen_dioxide": [25.0] * n_hours,
                "sulphur_dioxide": [10.0] * n_hours,
                "ozone": [45.0] * n_hours,
                "us_aqi": [70] * n_hours,
            }
        }

        city_configs = [
            {"id": "karachi", "name": "Karachi", "latitude": 24.86, "longitude": 67.00},
        ]

        result = run_historical_ingestion(
            city_configs=city_configs,
            start_date="2023-06-01",
            end_date="2023-06-02",
            save=False,
        )

        df = result["dataframe"]
        assert not df.empty
        assert "aqi" in df.columns
        assert "pm25" in df.columns
        assert "temperature" in df.columns
        assert result["validation"]["status"] in ("PASS", "WARNING")
