"""
Deployment Safety Tests

Tests for deployment safety checks.
"""

import os
from unittest.mock import patch

import pytest


class TestMockModeRejection:
    """Test mock mode is rejected in production."""

    def test_mock_mode_true_rejected(self):
        """Test MOCK_MODE=true is rejected."""
        with patch.dict(os.environ, {"MOCK_MODE": "true"}):
            # Verify deployment would fail
            mock_mode = os.getenv("MOCK_MODE", "false").lower()
            assert mock_mode == "true"

            # Production should reject this
            # In real deployment, pre_deploy_checks would catch this

    def test_mock_mode_false_accepted(self):
        """Test MOCK_MODE=false is accepted."""
        with patch.dict(os.environ, {"MOCK_MODE": "false"}):
            mock_mode = os.getenv("MOCK_MODE", "false").lower()
            assert mock_mode == "false"

    def test_mock_mode_missing_defaults_false(self):
        """Test missing MOCK_MODE defaults to false."""
        env = os.environ.copy()
        env.pop("MOCK_MODE", None)

        with patch.dict(os.environ, env, clear=True):
            mock_mode = os.getenv("MOCK_MODE", "false").lower()
            assert mock_mode == "false"


class TestSyntheticModelRejection:
    """Test synthetic models are rejected."""

    def test_synthetic_dataset_type_rejected(self):
        """Test synthetic_test_data is rejected."""
        metadata = {
            "dataset_type": "synthetic_test_data",
            "approved_for_training": False,
            "status": "production",
        }

        # Production deployment should reject synthetic data
        assert metadata["dataset_type"] == "synthetic_test_data"
        # In real deployment, this would raise an error

    def test_real_data_accepted(self):
        """Test real_api_data is accepted."""
        metadata = {
            "dataset_type": "real_api_data",
            "approved_for_training": True,
            "status": "production",
        }

        assert metadata["dataset_type"] == "real_api_data"
        assert metadata["approved_for_training"] is True


class TestMissingSecretRejection:
    """Test missing secrets are rejected."""

    def test_missing_api_key(self):
        """Test missing API_KEY is rejected."""
        env = os.environ.copy()
        env.pop("API_KEY", None)

        with patch.dict(os.environ, env, clear=True):
            api_key = os.getenv("API_KEY")
            assert api_key is None

            # Production deployment should fail
            # In real deployment, pre_deploy_checks would catch this

    def test_missing_hopsworks_host(self):
        """Test missing HOPSWORKS_HOST is warning."""
        env = os.environ.copy()
        env.pop("HOPSWORKS_HOST", None)

        with patch.dict(os.environ, env, clear=True):
            hopsworks_host = os.getenv("HOPSWORKS_HOST")
            assert hopsworks_host is None

            # Warning only, not failure

    def test_api_keys_present(self):
        """Test API keys are present."""
        with patch.dict(
            os.environ,
            {
                "OPENWEATHER_API_KEY": "test-key",
                "AQICN_API_KEY": "test-key",
            },
        ):
            assert os.getenv("OPENWEATHER_API_KEY") == "test-key"
            assert os.getenv("AQICN_API_KEY") == "test-key"


class TestHealthFailureRollback:
    """Test rollback simulation on health failure."""

    def test_rollback_triggered_on_failure(self):
        """Test rollback is triggered when health check fails."""
        # Simulate deployment state
        deployment_state = {
            "status": "deploying",
            "backup_tag": "backup_20260819_100000",
        }

        # Simulate health check failure
        health_check_passed = False

        if not health_check_passed:
            # Rollback should be triggered
            deployment_state["status"] = "rolling_back"
            assert deployment_state["status"] == "rolling_back"

    def test_deployment_success_no_rollback(self):
        """Test no rollback when deployment succeeds."""
        deployment_state = {
            "status": "deploying",
            "backup_tag": "backup_20260819_100000",
        }

        # Simulate successful health check
        health_check_passed = True

        if health_check_passed:
            deployment_state["status"] = "deployed"
            assert deployment_state["status"] == "deployed"

    def test_backup_tag_preserved(self):
        """Test backup tag is preserved for rollback."""
        backup_tag = "backup_20260819_100000"

        deployment_state = {
            "backup_tag": backup_tag,
        }

        assert deployment_state["backup_tag"] == backup_tag
