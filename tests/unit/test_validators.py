"""
Tests for data validators.

Tests cover:
- Schema validation (column presence)
- Staleness detection
- Duplicate detection and removal
- Missing value reporting
- Full validation pipeline
"""

import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone

from src.data.validators import (
    validate_schema,
    check_staleness,
    check_duplicates,
    check_missing_values,
    drop_duplicates,
    full_validation,
    REQUIRED_COLUMNS,
)
from src.data.schemas import ValidationStatus


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def valid_dataframe():
    """DataFrame matching StandardObservation schema."""
    now = datetime.now(timezone.utc)
    return pd.DataFrame(
        {
            "timestamp": [now - timedelta(hours=i) for i in range(5)],
            "location_id": ["karachi"] * 5,
            "city_name": ["Karachi"] * 5,
            "temperature": [34.0, 34.5, 35.0, 34.8, 34.2],
            "humidity": [70.0, 72.0, 68.0, 71.0, 73.0],
            "aqi": [150, 160, 155, 165, 158],
            "data_source": ["openweather"] * 5,
        }
    )


@pytest.fixture
def dataframe_with_duplicates():
    """DataFrame with duplicate (timestamp, location_id) pairs."""
    now = datetime.now(timezone.utc)
    return pd.DataFrame(
        {
            "timestamp": [
                now,
                now,  # Duplicate
                now - timedelta(hours=1),
                now - timedelta(hours=1),  # Duplicate
                now - timedelta(hours=2),
            ],
            "location_id": ["karachi"] * 5,
            "temperature": [34.0, 34.5, 35.0, 34.8, 34.2],
            "data_source": ["openweather"] * 5,
        }
    )


@pytest.fixture
def dataframe_with_missing():
    """DataFrame with missing values."""
    now = datetime.now(timezone.utc)
    return pd.DataFrame(
        {
            "timestamp": [now, now, now],
            "location_id": ["karachi", "lahore", "islamabad"],
            "temperature": [34.0, None, 36.0],
            "humidity": [70.0, 72.0, None],
            "aqi": [150, None, None],
            "data_source": ["openweather"] * 3,
        }
    )


@pytest.fixture
def empty_dataframe():
    """Empty DataFrame."""
    return pd.DataFrame(columns=["timestamp", "location_id", "temperature", "data_source"])


# =============================================================================
# Test Schema Validation
# =============================================================================


class TestValidateSchema:
    """Tests for validate_schema function."""

    def test_valid_schema(self, valid_dataframe):
        """DataFrame with all required columns passes."""
        errors = validate_schema(valid_dataframe)
        assert len(errors) == 0

    def test_missing_required_column(self):
        """DataFrame missing required column fails."""
        df = pd.DataFrame({"temperature": [34.0], "humidity": [70.0]})
        errors = validate_schema(df)
        assert len(errors) == 2
        assert any("timestamp" in e for e in errors)
        assert any("location_id" in e for e in errors)

    def test_custom_required_columns(self, valid_dataframe):
        """Custom required columns are checked."""
        errors = validate_schema(valid_dataframe, required_columns=["temperature"])
        assert len(errors) == 0

    def test_empty_dataframe(self, empty_dataframe):
        """Empty DataFrame with correct columns passes."""
        errors = validate_schema(empty_dataframe)
        assert len(errors) == 0


# =============================================================================
# Test Staleness Detection
# =============================================================================


class TestCheckStaleness:
    """Tests for check_staleness function."""

    def test_fresh_data(self):
        """Fresh data returns None."""
        now = datetime.now(timezone.utc)
        df = pd.DataFrame({"timestamp": [now]})
        result = check_staleness(df, max_age_hours=2.0)
        assert result is None

    def test_stale_data(self):
        """Stale data returns warning string."""
        old_time = datetime.now(timezone.utc) - timedelta(hours=5)
        df = pd.DataFrame({"timestamp": [old_time]})
        result = check_staleness(df, max_age_hours=2.0)
        assert result is not None
        assert "stale" in result.lower()

    def test_empty_dataframe(self, empty_dataframe):
        """Empty DataFrame returns None."""
        result = check_staleness(empty_dataframe)
        assert result is None

    def test_missing_timestamp_column(self):
        """DataFrame without timestamp column returns None."""
        df = pd.DataFrame({"temperature": [34.0]})
        result = check_staleness(df)
        assert result is None


# =============================================================================
# Test Duplicate Detection
# =============================================================================


class TestCheckDuplicates:
    """Tests for check_duplicates function."""

    def test_no_duplicates(self, valid_dataframe):
        """No duplicates returns 0."""
        count = check_duplicates(valid_dataframe)
        assert count == 0

    def test_with_duplicates(self, dataframe_with_duplicates):
        """Duplicates are counted."""
        count = check_duplicates(dataframe_with_duplicates)
        assert count == 2

    def test_custom_key_columns(self):
        """Custom key columns are used."""
        df = pd.DataFrame(
            {
                "timestamp": [datetime.now(timezone.utc)] * 3,
                "location_id": ["karachi", "karachi", "lahore"],
                "city_name": ["Karachi", "Karachi", "Lahore"],
            }
        )
        count = check_duplicates(df, key_columns=["location_id", "city_name"])
        assert count == 1


class TestDropDuplicates:
    """Tests for drop_duplicates function."""

    def test_removes_duplicates(self, dataframe_with_duplicates):
        """Duplicates are removed, keeping first occurrence."""
        cleaned = drop_duplicates(dataframe_with_duplicates)
        assert len(cleaned) == 3

    def test_no_duplicates_unchanged(self, valid_dataframe):
        """DataFrame without duplicates is unchanged."""
        cleaned = drop_duplicates(valid_dataframe)
        assert len(cleaned) == 5


# =============================================================================
# Test Missing Values
# =============================================================================


class TestCheckMissingValues:
    """Tests for check_missing_values function."""

    def test_no_missing(self, valid_dataframe):
        """No missing values returns empty dict."""
        result = check_missing_values(valid_dataframe)
        assert len(result) == 0

    def test_with_missing(self, dataframe_with_missing):
        """Missing values are counted."""
        result = check_missing_values(dataframe_with_missing)
        assert "temperature" in result
        assert result["temperature"] == 1
        assert "humidity" in result
        assert result["humidity"] == 1
        assert "aqi" in result
        assert result["aqi"] == 2


# =============================================================================
# Test Full Validation
# =============================================================================


class TestFullValidation:
    """Tests for full_validation pipeline."""

    def test_valid_data(self, valid_dataframe):
        """Valid data passes all checks."""
        report = full_validation(valid_dataframe)
        assert report.status == ValidationStatus.PASS
        assert report.total_records == 5
        assert report.duplicate_count == 0
        assert len(report.errors) == 0

    def test_data_with_duplicates(self, dataframe_with_duplicates):
        """Data with duplicates gets warning status."""
        report = full_validation(dataframe_with_duplicates)
        assert report.status == ValidationStatus.WARNING
        assert report.duplicate_count == 2

    def test_stale_data(self):
        """Stale data gets warning status."""
        old_time = datetime.now(timezone.utc) - timedelta(hours=5)
        df = pd.DataFrame(
            {
                "timestamp": [old_time],
                "location_id": ["karachi"],
                "data_source": ["openweather"],
            }
        )
        report = full_validation(df, max_staleness_hours=2.0)
        assert report.status == ValidationStatus.WARNING
        assert report.staleness_hours is not None
        assert report.staleness_hours > 4.0

    def test_empty_data(self, empty_dataframe):
        """Empty data with correct columns passes validation."""
        report = full_validation(empty_dataframe)
        assert report.status == ValidationStatus.PASS
        assert report.total_records == 0

    def test_missing_required_columns(self):
        """Missing required columns fail validation."""
        df = pd.DataFrame({"temperature": [34.0]})
        report = full_validation(df)
        assert report.status == ValidationStatus.FAIL
        assert len(report.errors) > 0

    def test_report_structure(self, valid_dataframe):
        """Report has correct structure."""
        report = full_validation(valid_dataframe)
        assert hasattr(report, "status")
        assert hasattr(report, "total_records")
        assert hasattr(report, "missing_values")
        assert hasattr(report, "duplicate_count")
        assert hasattr(report, "warnings")
        assert hasattr(report, "errors")
