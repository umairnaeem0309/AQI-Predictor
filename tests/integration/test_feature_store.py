"""
Integration test for feature store — end-to-end local store operations.

Tests the complete flow: feature engineering → feature store insert → retrieve → validate.
Uses local store only (Hopsworks integration requires credentials).
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.feature_store.local_store import LocalStore
from src.feature_store.schemas import (
    DatasetMetadata,
    DatasetType,
    get_feature_group_name,
)
from src.features.feature_engineering import engineer_features


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_store():
    """Temporary local feature store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LocalStore(base_path=Path(tmpdir))
        store.connect()
        yield store
        store.close()


@pytest.fixture
def feature_engineered_data():
    """Feature-engineered dataset for 3 cities, 3 days."""
    base_time = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
    rows = []
    for city_id, temp_base, aqi_base in [("karachi", 30, 100), ("lahore", 35, 160), ("islamabad", 28, 70)]:
        for i in range(72):  # 3 days
            ts = base_time + timedelta(hours=i)
            rows.append({
                "timestamp": ts,
                "location_id": city_id,
                "city_name": city_id.title(),
                "temperature": temp_base + np.sin(i / 24 * 2 * np.pi) * 3 + np.random.randn(),
                "humidity": 60 + np.random.randn() * 5,
                "wind_speed": 3 + np.random.rand() * 3,
                "pressure": 1010 + np.random.randn() * 2,
                "aqi": aqi_base + np.sin(i / 48 * 2 * np.pi) * 20 + np.random.randn() * 5,
                "pm25": 30 + np.random.rand() * 20,
                "pm10": 50 + np.random.rand() * 25,
                "co": 200 + np.random.rand() * 50,
                "no2": 20 + np.random.rand() * 10,
                "so2": 10 + np.random.rand() * 5,
                "o3": 40 + np.random.rand() * 15,
                "weather_condition": "clear",
                "data_source": "openweather",
            })
    df = pd.DataFrame(rows)
    return engineer_features(df)


@pytest.fixture
def synthetic_metadata():
    """Synthetic test metadata."""
    return DatasetMetadata(
        dataset_version="v20260808_integration_test",
        dataset_type=DatasetType.SYNTHETIC_TEST,
        approved_for_training=False,
        approved_for_evaluation=False,
        source="synthetic",
    )


# =============================================================================
# Integration Tests
# =============================================================================


class TestFeatureStoreEndToEnd:
    """End-to-end feature store tests."""

    def test_insert_and_retrieve_features(self, temp_store, feature_engineered_data, synthetic_metadata):
        """Features can be inserted and retrieved."""
        # Insert
        result = temp_store.insert_features(
            "aqi_features_test",
            feature_engineered_data,
            synthetic_metadata,
        )
        assert result is True

        # Retrieve
        retrieved = temp_store.get_features("aqi_features_test")
        assert len(retrieved) == len(feature_engineered_data)
        assert "aqi" in retrieved.columns
        assert "temperature" in retrieved.columns

    def test_feature_count_matches(self, temp_store, feature_engineered_data, synthetic_metadata):
        """Retrieved features have correct column count."""
        temp_store.insert_features(
            "aqi_features_test",
            feature_engineered_data,
            synthetic_metadata,
        )
        retrieved = temp_store.get_features("aqi_features_test")
        assert len(retrieved.columns) >= 20  # At least 20 features

    def test_multi_city_features(self, temp_store, feature_engineered_data, synthetic_metadata):
        """All cities are stored correctly."""
        temp_store.insert_features(
            "aqi_features_test",
            feature_engineered_data,
            synthetic_metadata,
        )
        retrieved = temp_store.get_features("aqi_features_test")
        assert retrieved["location_id"].nunique() == 3

    def test_lineage_metadata_complete(self, temp_store, feature_engineered_data, synthetic_metadata):
        """Lineage metadata includes all required fields."""
        temp_store.insert_features(
            "aqi_features_test",
            feature_engineered_data,
            synthetic_metadata,
        )
        meta = temp_store.get_metadata("aqi_features_test")
        assert meta is not None
        lineage = meta["lineage"]
        assert "feature_version" in lineage
        assert "schema_version" in lineage
        assert "source_dataset_version" in lineage
        assert "creation_timestamp" in lineage
        assert "dataset_type" in lineage

    def test_dataset_type_safety(self, temp_store, feature_engineered_data, synthetic_metadata):
        """Synthetic data is blocked from production groups."""
        with pytest.raises(ValueError, match="synthetic test data"):
            temp_store.insert_features(
                "aqi_features_prod",
                feature_engineered_data,
                synthetic_metadata,
            )

    def test_feature_group_naming(self):
        """Feature group naming follows convention."""
        test_name = get_feature_group_name("aqi_features", DatasetType.SYNTHETIC_TEST)
        prod_name = get_feature_group_name("aqi_features", DatasetType.REAL_TRAINING)
        assert test_name == "aqi_features_test"
        assert prod_name == "aqi_features_prod"

    def test_delete_and_verify(self, temp_store, feature_engineered_data, synthetic_metadata):
        """Feature group can be deleted."""
        temp_store.insert_features(
            "aqi_features_test",
            feature_engineered_data,
            synthetic_metadata,
        )
        temp_store.delete_feature_group("aqi_features_test")
        retrieved = temp_store.get_features("aqi_features_test")
        assert retrieved.empty

    def test_multiple_versions(self, temp_store, feature_engineered_data, synthetic_metadata):
        """Multiple versions can coexist."""
        temp_store.insert_features(
            "aqi_features_test",
            feature_engineered_data,
            synthetic_metadata,
            version=1,
        )
        temp_store.insert_features(
            "aqi_features_test",
            feature_engineered_data,
            synthetic_metadata,
            version=2,
        )
        v1 = temp_store.get_features("aqi_features_test", version=1)
        v2 = temp_store.get_features("aqi_features_test", version=2)
        assert len(v1) == len(v2)
