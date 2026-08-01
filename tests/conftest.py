"""
Shared pytest fixtures for AQI Predictor test suite.
"""

import sys
from pathlib import Path

import pytest

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def project_root():
    """Return the project root directory path."""
    return PROJECT_ROOT


@pytest.fixture
def config_path():
    """Return the path to config.yaml."""
    return PROJECT_ROOT / "config.yaml"
