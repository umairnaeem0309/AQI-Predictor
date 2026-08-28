"""
Environment Verification Tests.

Verifies that the development environment is correctly configured:
- Python version
- Critical library imports
- Configuration loading
- Project module imports
- Logging initialization
"""

import logging
import sys
from pathlib import Path

import pytest


class TestPythonVersion:
    """Verify Python version requirements."""

    def test_python_version_is_3_11(self):
        """Python 3.11 is required for Hopsworks compatibility."""
        assert sys.version_info[:2] == (
            3,
            11,
        ), f"Expected Python 3.11, got {sys.version_info.major}.{sys.version_info.minor}"

    def test_python_version_not_3_12_plus(self):
        """Python 3.12+ is forbidden due to imp module removal."""
        assert sys.version_info.major < 3 or (
            sys.version_info.major == 3 and sys.version_info.minor < 12
        ), "Python 3.12+ is not supported (Hopsworks incompatibility)"


class TestCriticalImports:
    """Verify all critical libraries can be imported."""

    def test_import_sklearn(self):
        import sklearn

        assert sklearn.__version__ is not None

    def test_import_xgboost(self):
        import xgboost

        assert xgboost.__version__ is not None

    def test_import_pandas(self):
        import pandas as pd

        assert pd.__version__ is not None

    def test_import_numpy(self):
        import numpy as np

        assert np.__version__ is not None

    def test_import_fastapi(self):
        import fastapi

        assert fastapi.__version__ is not None

    def test_import_streamlit(self):
        import streamlit

        assert streamlit.__version__ is not None

    def test_import_pydantic(self):
        import pydantic

        assert pydantic.__version__ is not None

    def test_import_yaml(self):
        import yaml

        assert yaml.__version__ is not None

    def test_import_dotenv(self):
        from importlib.metadata import version as get_version

        pkg_version = get_version("python-dotenv")
        assert pkg_version is not None

    def test_import_requests(self):
        import requests

        assert requests.__version__ is not None

    def test_import_duckdb(self):
        import duckdb

        assert duckdb.__version__ is not None


class TestTensorflowImport:
    """TensorFlow import test — may be slow or unavailable."""

    def test_import_tensorflow(self):
        """Verify TensorFlow can be imported.

        Note: tensorflow-cpu is preferred. If import fails, check
        that tensorflow-cpu is installed (not full tensorflow).
        """
        try:
            import tensorflow as tf

            assert tf.__version__ is not None
        except ImportError:
            pytest.skip("TensorFlow not installed — install tensorflow-cpu")


class TestConfigLoading:
    """Verify configuration system works correctly."""

    def test_config_yaml_exists(self):
        """config.yaml must exist at project root."""
        config_path = Path(__file__).resolve().parent.parent.parent / "config.yaml"
        assert config_path.exists(), f"config.yaml not found at {config_path}"

    def test_config_loads_successfully(self):
        """config.yaml must load without errors."""
        from src.config import load_config

        config = load_config()
        assert isinstance(config, dict)
        assert "project" in config
        assert "cities" in config

    def test_config_has_cities(self):
        """Configuration must include city definitions."""
        from src.config import load_config

        config = load_config()
        cities = config.get("cities", [])
        assert len(cities) == 3, f"Expected 3 cities, got {len(cities)}"
        city_ids = [c["id"] for c in cities]
        assert "karachi" in city_ids
        assert "lahore" in city_ids
        assert "islamabad" in city_ids

    def test_config_no_hardcoded_hopsworks_host(self):
        """Hopsworks host must not be hardcoded in config.yaml."""
        from src.config import load_config

        config = load_config()
        hopsworks_config = config.get("feature_store", {})
        # Host should not be in config — it comes from env var HOPSWORKS_HOST
        assert (
            "host" not in hopsworks_config
        ), "Hopsworks host must be in environment variable, not config.yaml"


class TestProjectModuleImports:
    """Verify project modules can be imported."""

    def test_import_config_module(self):
        from src.config import get_env, load_config, setup_logging

        assert callable(load_config)
        assert callable(setup_logging)
        assert callable(get_env)


class TestLoggingInitialization:
    """Verify logging system initializes correctly."""

    def test_setup_logging_runs_without_error(self):
        """setup_logging() must execute without raising exceptions."""
        from src.config import setup_logging

        setup_logging()

    def test_logging_level_is_configurable(self):
        """Logging level must be set after setup_logging()."""
        from src.config import setup_logging

        setup_logging()
        root_logger = logging.getLogger()
        assert root_logger.level <= logging.INFO, f"Expected INFO or lower, got {root_logger.level}"

    def test_logger_can_be_created(self):
        """Modules should be able to create named loggers."""
        logger = logging.getLogger("test_aqi_predictor")
        assert logger is not None
        logger.info("Test logger created successfully")


class TestProjectStructure:
    """Verify essential project directories exist."""

    def test_src_directory_exists(self):
        src_path = Path(__file__).resolve().parent.parent.parent / "src"
        assert src_path.exists()

    def test_app_directory_exists(self):
        app_path = Path(__file__).resolve().parent.parent.parent / "app"
        assert app_path.exists()

    def test_data_directories_exist(self):
        project_root = Path(__file__).resolve().parent.parent.parent
        for subdir in ["raw", "processed", "mock"]:
            data_path = project_root / "data" / subdir
            assert data_path.exists(), f"Directory data/{subdir} not found"

    def test_pipelines_directory_exists(self):
        pipelines_path = Path(__file__).resolve().parent.parent.parent / "pipelines"
        assert pipelines_path.exists()

    def test_docs_directory_exists(self):
        docs_path = Path(__file__).resolve().parent.parent.parent / "docs"
        assert docs_path.exists()
