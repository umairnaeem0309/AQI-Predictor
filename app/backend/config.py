"""
Backend Configuration

Configuration for FastAPI backend including rate limiting,
feature store settings, and security.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class BackendConfig:
    """Backend configuration settings."""

    # Application
    app_name: str = "AQI Predictor API"
    app_version: str = "1.0.0"
    debug: bool = False

    # API Security
    api_key_header: str = "X-API-Key"
    api_key: Optional[str] = None

    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 1000  # requests per window
    rate_limit_window_seconds: int = 60  # window duration

    # Feature Store
    feature_store_primary: str = "hopsworks"  # hopsworks or local
    feature_store_local_fallback_enabled: bool = False  # Only for development
    hopsworks_host: Optional[str] = None

    # Model
    model_load_timeout_seconds: int = 30
    require_production_model: bool = True  # Never fallback to mock

    # CORS
    cors_origins: list = None
    cors_methods: list = None

    # Logging
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "BackendConfig":
        """Load configuration from environment variables."""
        return cls(
            api_key=os.getenv("API_KEY"),
            hopsworks_host=os.getenv("HOPSWORKS_HOST"),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            rate_limit_enabled=os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true",
            rate_limit_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "100")),
            rate_limit_window_seconds=int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")),
            feature_store_primary=os.getenv("FEATURE_STORE_PRIMARY", "hopsworks"),
            feature_store_local_fallback_enabled=os.getenv(
                "FEATURE_STORE_LOCAL_FALLBACK", "false"
            ).lower()
            == "true",
            cors_origins=os.getenv("CORS_ORIGINS", "*").split(","),
            cors_methods=os.getenv("CORS_METHODS", "GET,POST").split(","),
        )

    def validate(self):
        """Validate configuration."""
        errors = []

        if self.require_production_model and not self.api_key:
            # API key is optional for development
            pass

        if self.feature_store_primary == "hopsworks" and not self.hopsworks_host:
            if not self.feature_store_local_fallback_enabled:
                errors.append(
                    "HOPSWORKS_HOST required when feature_store_primary=hopsworks "
                    "and local fallback is disabled"
                )

        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")


# Default configuration instance
default_config = BackendConfig.from_env()
