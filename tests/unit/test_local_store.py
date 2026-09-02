"""
Tests for Local Feature Store — DuckDB + Parquet implementation.
"""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_store.local_store import LocalStore
from src.feature_store.schemas import DatasetMetadata, DatasetType, get_feature_group_name

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_store():
    """Create a temporary local feature store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LocalStore(base_path=Path(tmpdir))
        store.connect()
        yield store
        store.close()


@pytest.fixture
def sample_features():
    """Sample feature DataFrame."""
    return pd.DataFrame(
        {
            "location_id": ["karachi"] * 5,
            "timestamp": pd.date_range("2026-08-01", periods=5, freq="h", tz="UTC"),
            "city_name": ["Karachi"] * 5,
            "temperature": [30.0, 31.0, 32.0, 33.0, 34.0],
            "humidity": [60.0, 62.0, 64.0, 66.0, 68.0],
            "aqi": [100, 105, 110, 115, 120],
            "pm25": [40.0, 42.0, 44.0, 46.0, 48.0],
        }
    )


@pytest.fixture
def sample_targets():
    """Sample target DataFrame."""
    return pd.DataFrame(
        {
            "location_id": ["karachi"] * 5,
            "timestamp": pd.date_range("2026-08-01", periods=5, freq="h", tz="UTC"),
            "target_aqi_24h": [110, 115, 120, 125, 130],
            "target_aqi_48h": [120, 125, 130, 135, 140],
            "target_aqi_72h": [130, 135, 140, 145, 150],
        }
    )


@pytest.fixture
def synthetic_metadata():
    """Metadata for synthetic test data."""
    return DatasetMetadata(
        dataset_version="v20260808_test",
        dataset_type=DatasetType.SYNTHETIC_TEST,
        approved_for_training=False,
        approved_for_evaluation=False,
        source="synthetic",
    )


@pytest.fixture
def real_metadata():
    """Metadata for real production data."""
    return DatasetMetadata(
        dataset_version="v20260808_prod",
        dataset_type=DatasetType.REAL_TRAINING,
        approved_for_training=True,
        approved_for_evaluation=True,
        source="openweather",
    )


# =============================================================================
# Test Insert Operations
# =============================================================================


class TestInsertOperations:
    """Tests for feature and target insertion."""

    def test_insert_features(self, temp_store, sample_features, synthetic_metadata):
        """Features are inserted successfully."""
        result = temp_store.insert_features(
            "aqi_features_test",
            sample_features,
            synthetic_metadata,
        )
        assert result is True

    def test_insert_targets(self, temp_store, sample_targets, synthetic_metadata):
        """Targets are inserted successfully."""
        result = temp_store.insert_targets(
            "aqi_targets_test",
            sample_targets,
            synthetic_metadata,
        )
        assert result is True

    def test_insert_creates_parquet(self, temp_store, sample_features, synthetic_metadata):
        """Insert creates a Parquet file."""
        temp_store.insert_features(
            "aqi_features_test",
            sample_features,
            synthetic_metadata,
        )
        parquet_path = temp_store._get_parquet_path("aqi_features_test", 1)
        assert parquet_path.exists()

    def test_insert_creates_metadata(self, temp_store, sample_features, synthetic_metadata):
        """Insert creates a metadata file."""
        temp_store.insert_features(
            "aqi_features_test",
            sample_features,
            synthetic_metadata,
        )
        meta = temp_store.get_metadata("aqi_features_test", 1)
        assert meta is not None
        assert meta["group_name"] == "aqi_features_test"
        assert meta["record_count"] == 5


# =============================================================================
# Test Retrieve Operations
# =============================================================================


class TestRetrieveOperations:
    """Tests for feature and target retrieval."""

    def test_get_features(self, temp_store, sample_features, synthetic_metadata):
        """Features are retrieved correctly."""
        temp_store.insert_features(
            "aqi_features_test",
            sample_features,
            synthetic_metadata,
        )
        retrieved = temp_store.get_features("aqi_features_test")
        assert len(retrieved) == 5
        assert "temperature" in retrieved.columns
        assert retrieved["aqi"].iloc[0] == 100

    def test_get_features_with_columns(self, temp_store, sample_features, synthetic_metadata):
        """Specific columns can be retrieved."""
        temp_store.insert_features(
            "aqi_features_test",
            sample_features,
            synthetic_metadata,
        )
        retrieved = temp_store.get_features(
            "aqi_features_test",
            columns=["location_id", "aqi"],
        )
        assert list(retrieved.columns) == ["location_id", "aqi"]

    def test_get_features_nonexistent(self, temp_store):
        """Nonexistent feature group returns empty DataFrame."""
        retrieved = temp_store.get_features("nonexistent_group")
        assert retrieved.empty

    def test_get_targets(self, temp_store, sample_targets, synthetic_metadata):
        """Targets are retrieved correctly."""
        temp_store.insert_targets(
            "aqi_targets_test",
            sample_targets,
            synthetic_metadata,
        )
        retrieved = temp_store.get_targets("aqi_targets_test")
        assert len(retrieved) == 5
        assert "target_aqi_24h" in retrieved.columns


# =============================================================================
# Test Delete Operations
# =============================================================================


class TestDeleteOperations:
    """Tests for feature group deletion."""

    def test_delete_feature_group(self, temp_store, sample_features, synthetic_metadata):
        """Feature group is deleted successfully."""
        temp_store.insert_features(
            "aqi_features_test",
            sample_features,
            synthetic_metadata,
        )
        result = temp_store.delete_feature_group("aqi_features_test", 1)
        assert result is True

        # Verify deletion
        retrieved = temp_store.get_features("aqi_features_test")
        assert retrieved.empty


# =============================================================================
# Test Metadata Management
# =============================================================================


class TestMetadataManagement:
    """Tests for metadata storage and retrieval."""

    def test_metadata_record_count(self, temp_store, sample_features, synthetic_metadata):
        """Metadata records correct count."""
        temp_store.insert_features(
            "aqi_features_test",
            sample_features,
            synthetic_metadata,
        )
        meta = temp_store.get_metadata("aqi_features_test", 1)
        assert meta["record_count"] == 5

    def test_metadata_dataset_type(self, temp_store, sample_features, synthetic_metadata):
        """Metadata records dataset type."""
        temp_store.insert_features(
            "aqi_features_test",
            sample_features,
            synthetic_metadata,
        )
        meta = temp_store.get_metadata("aqi_features_test", 1)
        assert meta["dataset_metadata"]["dataset_type"] == "synthetic_test_data"

    def test_metadata_lineage(self, temp_store, sample_features, synthetic_metadata):
        """Metadata includes lineage information."""
        temp_store.insert_features(
            "aqi_features_test",
            sample_features,
            synthetic_metadata,
        )
        meta = temp_store.get_metadata("aqi_features_test", 1)
        assert "lineage" in meta
        assert "feature_version" in meta["lineage"]
        assert "source_dataset_version" in meta["lineage"]
        assert "creation_timestamp" in meta["lineage"]


# =============================================================================
# Test List Feature Groups
# =============================================================================


class TestListFeatureGroups:
    """Tests for listing feature groups."""

    def test_list_empty(self, temp_store):
        """Empty store returns empty list."""
        groups = temp_store.list_feature_groups()
        assert groups == []

    def test_list_after_insert(self, temp_store, sample_features, synthetic_metadata):
        """Inserted groups appear in list."""
        temp_store.insert_features(
            "aqi_features_test",
            sample_features,
            synthetic_metadata,
        )
        groups = temp_store.list_feature_groups()
        assert "aqi_features_test" in groups


# =============================================================================
# Test DuckDB Query
# =============================================================================


class TestDuckDBQuery:
    """Tests for DuckDB SQL queries."""

    def test_query_with_where(self, temp_store, sample_features, synthetic_metadata):
        """SQL WHERE clause filters results."""
        temp_store.insert_features(
            "aqi_features_test",
            sample_features,
            synthetic_metadata,
        )
        result = temp_store.query_features(
            "aqi_features_test",
            "aqi > 110",
        )
        assert len(result) == 2  # aqi values: 115, 120
        assert (result["aqi"] > 110).all()


# =============================================================================
# Test Dataset Type Safety
# =============================================================================


class TestDatasetTypeSafety:
    """Tests for synthetic data safety validation."""

    def test_synthetic_rejected_from_prod(self, temp_store, sample_features, synthetic_metadata):
        """Synthetic data is rejected from production feature groups."""
        with pytest.raises(ValueError, match="synthetic test data"):
            temp_store.insert_features(
                "aqi_features_prod",
                sample_features,
                synthetic_metadata,
            )

    def test_real_data_accepted_in_prod(self, temp_store, sample_features, real_metadata):
        """Real data is accepted in production feature groups."""
        result = temp_store.insert_features(
            "aqi_features_prod",
            sample_features,
            real_metadata,
        )
        assert result is True

    def test_synthetic_accepted_in_test(self, temp_store, sample_features, synthetic_metadata):
        """Synthetic data is accepted in test feature groups."""
        result = temp_store.insert_features(
            "aqi_features_test",
            sample_features,
            synthetic_metadata,
        )
        assert result is True


# =============================================================================
# Test Feature Group Naming
# =============================================================================


class TestFeatureGroupNaming:
    """Tests for feature group naming convention."""

    def test_test_suffix(self):
        """Test data gets _test suffix."""
        name = get_feature_group_name("aqi_features", DatasetType.SYNTHETIC_TEST)
        assert name == "aqi_features_test"

    def test_prod_suffix(self):
        """Production data gets _prod suffix."""
        name = get_feature_group_name("aqi_features", DatasetType.REAL_TRAINING)
        assert name == "aqi_features_prod"
