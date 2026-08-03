"""
Configuration management for AQI Predictor.

Handles loading of config.yaml, environment variables from .env,
logging initialization, and credential validation.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Default config path
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

logger = logging.getLogger(__name__)


def load_environment(env_path: Optional[Path] = None) -> None:
    """Load environment variables from .env file.

    Args:
        env_path: Path to .env file. Defaults to project root .env.
    """
    if env_path is None:
        env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.debug("Loaded environment from %s", env_path)
    else:
        logger.info(".env file not found at %s. Using system environment.", env_path)


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load project configuration from config.yaml.

    Args:
        config_path: Path to config.yaml. Defaults to project root config.yaml.

    Returns:
        Dictionary containing configuration values.
    """
    if config_path is None:
        config_path = CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def setup_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """Initialize application logging.

    Reads log level and format from config.yaml or uses sensible defaults.

    Args:
        config: Optional pre-loaded configuration dictionary.
    """
    if config is None:
        try:
            config = load_config()
        except FileNotFoundError:
            config = {}

    logging_config = config.get("logging", {})
    log_level = os.environ.get("LOG_LEVEL", logging_config.get("level", "INFO"))
    log_format = logging_config.get(
        "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    date_format = logging_config.get("date_format", "%Y-%m-%d %H:%M:%S")

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=log_format,
        datefmt=date_format,
        force=True,
    )


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get an environment variable value.

    Args:
        key: Environment variable name.
        default: Default value if not set.

    Returns:
        Environment variable value or default.
    """
    return os.environ.get(key, default)


def get_api_key(service: str) -> Optional[str]:
    """Get API key for a specific service.

    Args:
        service: Service name (openweather, aqicn, hopsworks).

    Returns:
        API key string or None if not set.
    """
    env_key = f"{service.upper()}_API_KEY"
    return os.environ.get(env_key)


def validate_credentials(required: Optional[List[str]] = None) -> Dict[str, bool]:
    """Validate which API credentials are available.

    Does NOT raise errors — returns availability status for each service.
    This allows the application to operate in mock-only mode when
    credentials are missing.

    Args:
        required: List of service names to check. If None, checks all known services.

    Returns:
        Dictionary mapping service name to availability (True/False).
    """
    if required is None:
        required = ["openweather", "aqicn", "hopsworks"]

    availability = {}
    for service in required:
        key = get_api_key(service)
        available = key is not None and len(key) > 0
        availability[service] = available

        if available:
            # Log only that key exists, never the actual key
            masked = key[:3] + "***" + key[-3:] if len(key) > 6 else "***"
            logger.info("Credentials loaded for %s: key=%s", service, masked)
        else:
            logger.warning(
                "No credentials for %s — that source will be unavailable", service
            )

    return availability


def mask_key(key: Optional[str]) -> str:
    """Mask an API key for safe logging.

    Args:
        key: API key to mask.

    Returns:
        Masked key string (e.g., 'abc***xyz').
    """
    if not key:
        return "***"
    if len(key) <= 6:
        return "***"
    return key[:3] + "***" + key[-3:]


# Module-level initialization
load_environment()
_config = load_config() if CONFIG_PATH.exists() else {}
