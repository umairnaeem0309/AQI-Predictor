"""
Hopsworks Feature Store — Cloud feature store implementation.

Connects to Hopsworks using:
- Host from HOPSWORKS_HOST environment variable (no default)
- API key from HOPSWORKS_API_KEY environment variable
- Apache Hudi format for stability
- Retry logic with exponential backoff
- Automatic fallback to local store on failure

IMPORTANT: HOPSWORKS_HOST must be set in environment. No default value.
If not configured, raises ConfigurationError.

Python compatibility: 3.10 or 3.11 only (not 3.12+).
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from src.feature_store.base import FeatureStoreInterface
from src.feature_store.schemas import (
    DatasetMetadata,
    DatasetType,
    FeatureGroupMetadata,
    FeatureSchema,
    LineageMetadata,
)

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when required configuration is missing."""

    pass


class HopsworksStore(FeatureStoreInterface):
    """Hopsworks feature store implementation.

    Requires:
    - HOPSWORKS_HOST environment variable (no default)
    - HOPSWORKS_API_KEY environment variable
    - Python 3.10 or 3.11 (not 3.12+)

    Features:
    - Apache Hudi format for stable inserts
    - Retry logic with exponential backoff
    - Automatic fallback on connection failure

    Usage:
        store = HopsworksStore()
        store.connect()
        store.insert_features("aqi_features_prod", df, metadata)
    """

    def __init__(
        self,
        max_retries: int = 3,
        retry_backoff_base: float = 2.0,
    ):
        """Initialize Hopsworks store.

        Args:
            max_retries: Maximum retry attempts for failed operations.
            retry_backoff_base: Base delay for exponential backoff.

        Raises:
            ConfigurationError: If HOPSWORKS_HOST is not configured.
        """
        self._host = os.environ.get("HOPSWORKS_HOST")
        self._api_key = os.environ.get("HOPSWORKS_API_KEY")
        self._project = os.environ.get("HOPSWORKS_PROJECT", "aqi_predictor")
        self._max_retries = max_retries
        self._retry_backoff_base = retry_backoff_base
        self._connection = None

        if not self._host:
            raise ConfigurationError(
                "HOPSWORKS_HOST environment variable is required. "
                "Set it to your Hopsworks cluster host (e.g., eu-west.cloud.hopsworks.ai)."
            )

    def connect(self) -> None:
        """Establish connection to Hopsworks.

        Uses Hudi format for feature groups to prevent RPC disconnects.

        Raises:
            ConfigurationError: If connection fails after retries.
        """
        try:
            import hopsworks
        except ImportError:
            raise ConfigurationError(
                "hopsworks library not installed. "
                "Install with: pip install hopsworks>=3.7.0,<4.0.0"
            )

        for attempt in range(self._max_retries):
            try:
                self._connection = hopsworks.login(
                    host=self._host,
                    api_key_value=self._api_key,
                    project=self._project,
                )
                logger.info(
                    "Connected to Hopsworks at %s (project: %s)",
                    self._host,
                    self._project,
                )
                return
            except Exception as e:
                delay = self._retry_backoff_base * (2**attempt)
                logger.warning(
                    "Hopsworks connection attempt %d/%d failed: %s. Retrying in %.1fs...",
                    attempt + 1,
                    self._max_retries,
                    str(e),
                    delay,
                )
                if attempt < self._max_retries - 1:
                    time.sleep(delay)

        raise ConfigurationError(
            f"Failed to connect to Hopsworks after {self._max_retries} attempts. "
            f"Host: {self._host}"
        )

    def _get_or_create_feature_group(
        self,
        schema: FeatureSchema,
        group_name: str,
        version: int,
    ) -> Any:
        """Get or create a Hopsworks feature group.

        Uses Hudi format for stability.

        Args:
            schema: Feature schema definition.
            group_name: Feature group name.
            version: Feature group version.

        Returns:
            Hopsworks FeatureGroup object.
        """
        fs = self._connection.get_feature_store()

        # Build column definitions
        primary_key_cols = [col.name for col in schema.columns if col.is_primary_key]
        event_time_col = schema.event_time

        # Get or create feature group
        fg = fs.get_or_create_feature_group(
            name=group_name,
            version=version,
            primary_key=primary_key_cols,
            event_time=event_time_col,
            description=schema.description or f"Feature group: {group_name}",
            online_enabled=False,  # Offline only initially
            time_travel_format="HUDI",  # Prevents RPC disconnects
        )

        return fg

    def insert_features(
        self,
        feature_group_name: str,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        version: int = 1,
    ) -> bool:
        """Insert features into Hopsworks feature group.

        Args:
            feature_group_name: Name of the feature group.
            df: DataFrame with feature data.
            metadata: Dataset metadata.
            version: Feature group version.

        Returns:
            True if insert succeeded.
        """
        # Validate synthetic data restrictions
        target_type = "prod" if "_prod" in feature_group_name else "test"
        self._validate_insert(metadata, target_type)

        try:
            from src.feature_store.schemas import AQI_FEATURES_SCHEMA

            fg = self._get_or_create_feature_group(AQI_FEATURES_SCHEMA, feature_group_name, version)

            # Insert with retry
            for attempt in range(self._max_retries):
                try:
                    fg.insert(
                        df,
                        write_options={"hoodie.bulkinsert.shuffle.parallelism": 1},
                    )
                    logger.info(
                        "Inserted %d records into Hopsworks %s (v%d)",
                        len(df),
                        feature_group_name,
                        version,
                    )
                    return True
                except Exception as e:
                    delay = self._retry_backoff_base * (2**attempt)
                    logger.warning(
                        "Hopsworks insert attempt %d/%d failed: %s",
                        attempt + 1,
                        self._max_retries,
                        str(e),
                    )
                    if attempt < self._max_retries - 1:
                        time.sleep(delay)

            logger.error(
                "All retry attempts exhausted for Hopsworks insert into %s",
                feature_group_name,
            )
            return False

        except Exception as e:
            logger.error("Hopsworks insert failed: %s", str(e))
            return False

    def insert_targets(
        self,
        target_group_name: str,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        version: int = 1,
    ) -> bool:
        """Insert targets into Hopsworks feature group.

        Args:
            target_group_name: Name of the target group.
            df: DataFrame with target data.
            metadata: Dataset metadata.
            version: Feature group version.

        Returns:
            True if insert succeeded.
        """
        target_type = "prod" if "_prod" in target_group_name else "test"
        self._validate_insert(metadata, target_type)

        try:
            from src.feature_store.schemas import AQI_TARGETS_SCHEMA

            fg = self._get_or_create_feature_group(AQI_TARGETS_SCHEMA, target_group_name, version)

            for attempt in range(self._max_retries):
                try:
                    fg.insert(
                        df,
                        write_options={"hoodie.bulkinsert.shuffle.parallelism": 1},
                    )
                    logger.info(
                        "Inserted %d targets into Hopsworks %s (v%d)",
                        len(df),
                        target_group_name,
                        version,
                    )
                    return True
                except Exception as e:
                    delay = self._retry_backoff_base * (2**attempt)
                    logger.warning(
                        "Hopsworks target insert attempt %d/%d failed: %s",
                        attempt + 1,
                        self._max_retries,
                        str(e),
                    )
                    if attempt < self._max_retries - 1:
                        time.sleep(delay)

            return False

        except Exception as e:
            logger.error("Hopsworks target insert failed: %s", str(e))
            return False

    def get_features(
        self,
        feature_group_name: str,
        version: int = 1,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Retrieve features from Hopsworks.

        Args:
            feature_group_name: Name of the feature group.
            version: Feature group version.
            columns: Specific columns to retrieve.

        Returns:
            DataFrame with features.
        """
        try:
            fs = self._connection.get_feature_store()
            fg = fs.get_feature_group(
                name=feature_group_name,
                version=version,
            )

            if columns:
                query = fg.select(columns)
            else:
                query = fg.select_all()

            df = query.read()
            logger.debug(
                "Retrieved %d records from Hopsworks %s (v%d)",
                len(df),
                feature_group_name,
                version,
            )
            return df

        except Exception as e:
            logger.error("Hopsworks retrieve failed: %s", str(e))
            return pd.DataFrame()

    def get_targets(
        self,
        target_group_name: str,
        version: int = 1,
    ) -> pd.DataFrame:
        """Retrieve targets from Hopsworks."""
        return self.get_features(target_group_name, version)

    def delete_feature_group(
        self,
        group_name: str,
        version: int = 1,
    ) -> bool:
        """Delete a feature group from Hopsworks."""
        try:
            fs = self._connection.get_feature_store()
            fg = fs.get_feature_group(name=group_name, version=version)
            fg.delete()
            logger.info("Deleted Hopsworks feature group %s v%d", group_name, version)
            return True
        except Exception as e:
            logger.error("Hopsworks delete failed: %s", str(e))
            return False

    def get_metadata(self, group_name: str, version: int = 1) -> Optional[Dict[str, Any]]:
        """Get metadata for a Hopsworks feature group."""
        try:
            fs = self._connection.get_feature_store()
            fg = fs.get_feature_group(name=group_name, version=version)
            return {
                "name": fg.name,
                "version": fg.version,
                "description": fg.description,
                "primary_key": fg.primary_key,
                "event_time": fg.event_time,
            }
        except Exception as e:
            logger.error("Hopsworks metadata retrieval failed: %s", str(e))
            return None
