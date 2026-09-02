"""
Tests for Hopsworks Feature Store — All tests mock Hopsworks library.

Real Hopsworks integration requires credentials and should remain optional.
These tests verify the store's logic without making real API calls.
"""

import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, PropertyMock, patch

import pandas as pd
import pytest

from src.feature_store.hopsworks_store import ConfigurationError, HopsworksStore
from src.feature_store.schemas import DatasetMetadata, DatasetType

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_env():
    """Set up mock environment variables."""
    with patch.dict(
        os.environ,
        {
            "HOPSWORKS_HOST": "eu-west.cloud.hopsworks.ai",
            "HOPSWORKS_API_KEY": "test-api-key",
            "HOPSWORKS_PROJECT": "test-project",
        },
    ):
        yield


@pytest.fixture
def sample_features():
    """Sample feature DataFrame."""
    return pd.DataFrame(
        {
            "location_id": ["karachi"] * 3,
            "timestamp": pd.date_range("2026-08-01", periods=3, freq="h", tz="UTC"),
            "temperature": [30.0, 31.0, 32.0],
            "aqi": [100, 105, 110],
        }
    )


@pytest.fixture
def synthetic_metadata():
    """Synthetic test metadata."""
    return DatasetMetadata(
        dataset_version="v20260808_test",
        dataset_type=DatasetType.SYNTHETIC_TEST,
        approved_for_training=False,
    )


@pytest.fixture
def real_metadata():
    """Real production metadata."""
    return DatasetMetadata(
        dataset_version="v20260808_prod",
        dataset_type=DatasetType.REAL_TRAINING,
        approved_for_training=True,
    )


# =============================================================================
# Test Configuration
# =============================================================================


class TestHopsworksConfiguration:
    """Tests for Hopsworks configuration validation."""

    def test_requires_host(self):
        """Raises ConfigurationError when HOPSWORKS_HOST is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigurationError, match="HOPSWORKS_HOST"):
                HopsworksStore()

    def test_accepts_host(self, mock_env):
        """Accepts valid HOPSWORKS_HOST."""
        store = HopsworksStore()
        assert store._host == "eu-west.cloud.hopsworks.ai"

    def test_uses_env_vars(self, mock_env):
        """Reads configuration from environment variables."""
        store = HopsworksStore()
        assert store._host == "eu-west.cloud.hopsworks.ai"
        assert store._api_key == "test-api-key"
        assert store._project == "test-project"


# =============================================================================
# Test Connection
# =============================================================================


class TestHopsworksConnection:
    """Tests for Hopsworks connection handling."""

    def test_connect_success(self, mock_env):
        """Successful connection."""
        mock_hops = MagicMock()
        mock_hops.login.return_value = MagicMock()
        with patch.dict("sys.modules", {"hopsworks": mock_hops}):
            store = HopsworksStore()
            store.connect()
            mock_hops.login.assert_called_once()

    def test_connect_retries_on_failure(self, mock_env):
        """Connection retries on failure."""
        mock_hops = MagicMock()
        mock_hops.login.side_effect = [
            Exception("Connection failed"),
            MagicMock(),  # Second attempt succeeds
        ]
        with patch.dict("sys.modules", {"hopsworks": mock_hops}):
            store = HopsworksStore(max_retries=2)
            store.connect()
            assert mock_hops.login.call_count == 2

    def test_connect_all_retries_fail(self, mock_env):
        """ConfigurationError when all retries fail."""
        mock_hops = MagicMock()
        mock_hops.login.side_effect = Exception("Connection failed")
        with patch.dict("sys.modules", {"hopsworks": mock_hops}):
            store = HopsworksStore(max_retries=2)
            with pytest.raises(ConfigurationError, match="Failed to connect"):
                store.connect()


# =============================================================================
# Test Insert Logic
# =============================================================================


class TestHopsworksInsert:
    """Tests for Hopsworks insert operations."""

    def test_synthetic_rejected_from_prod(self, mock_env, sample_features, synthetic_metadata):
        """Synthetic data is rejected from production groups."""
        mock_hops = MagicMock()
        with patch.dict("sys.modules", {"hopsworks": mock_hops}):
            store = HopsworksStore()
            store._connection = MagicMock()

            with pytest.raises(ValueError, match="synthetic test data"):
                store.insert_features(
                    "aqi_features_prod",
                    sample_features,
                    synthetic_metadata,
                )

    def test_real_data_accepted_in_prod(self, mock_env, sample_features, real_metadata):
        """Real data is accepted in production groups."""
        mock_hops = MagicMock()
        with patch.dict("sys.modules", {"hopsworks": mock_hops}):
            store = HopsworksStore()
            store._connection = MagicMock()

            # Mock the feature store
            mock_fg = MagicMock()
            store._connection.feature_store.get_or_create_feature_group.return_value = mock_fg

            result = store.insert_features(
                "aqi_features_prod",
                sample_features,
                real_metadata,
            )
            assert result is True


# =============================================================================
# Test Fallback Behavior
# =============================================================================


class TestHopsworksFallback:
    """Tests for Hopsworks fallback to local store."""

    def test_fallback_on_connection_failure(self, mock_env):
        """Falls back to local store when Hopsworks fails."""
        mock_hops = MagicMock()
        mock_hops.login.side_effect = Exception("Connection failed")
        with patch.dict("sys.modules", {"hopsworks": mock_hops}):
            from src.feature_store import get_feature_store

            store = get_feature_store()

            # Should get LocalStore as fallback
            from src.feature_store.local_store import LocalStore

            assert isinstance(store, LocalStore)
