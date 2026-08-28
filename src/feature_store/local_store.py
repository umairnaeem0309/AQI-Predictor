"""
Local Feature Store — DuckDB + Parquet fallback implementation.

Provides offline feature storage using:
- Parquet files for efficient columnar storage
- DuckDB for SQL-based queries
- JSON metadata files for lineage tracking

This is the fallback when Hopsworks is unavailable, and the primary
development storage during local development.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd

from src.config import PROJECT_ROOT
from src.feature_store.base import FeatureStoreInterface
from src.feature_store.schemas import (
    DatasetMetadata,
    DatasetType,
    FeatureGroupMetadata,
    FeatureSchema,
    LineageMetadata,
)

logger = logging.getLogger(__name__)

# Storage paths
FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"
METADATA_DIR = FEATURES_DIR / "metadata"


class LocalStore(FeatureStoreInterface):
    """Local feature store using DuckDB and Parquet.

    Features are stored as Parquet files with DuckDB providing
    SQL query capabilities. Metadata is stored as JSON files.

    Usage:
        store = LocalStore()
        store.connect()
        store.insert_features("aqi_features_test", df, metadata)
        features = store.get_features("aqi_features_test")
    """

    def __init__(self, base_path: Optional[Path] = None):
        """Initialize local feature store.

        Args:
            base_path: Base directory for feature storage.
                Defaults to data/processed/features/.
        """
        self.base_path = base_path or FEATURES_DIR
        self.metadata_path = self.base_path / "metadata"
        self._connection = None

    def connect(self) -> None:
        """Establish DuckDB connection and create directories."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        self._connection = duckdb.connect(":memory:")
        logger.info("Local feature store connected: %s", self.base_path)

    def _get_parquet_path(self, group_name: str, version: int) -> Path:
        """Get Parquet file path for a feature group."""
        group_dir = self.base_path / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        return group_dir / f"v{version}.parquet"

    def _get_metadata_path(self, group_name: str, version: int) -> Path:
        """Get metadata file path for a feature group."""
        return self.metadata_path / f"{group_name}_v{version}.json"

    def _save_metadata(
        self,
        group_name: str,
        version: int,
        metadata: DatasetMetadata,
        lineage: LineageMetadata,
        record_count: int,
    ) -> None:
        """Save feature group metadata to JSON."""
        meta = {
            "group_name": group_name,
            "version": version,
            "dataset_metadata": metadata.model_dump(),
            "lineage": lineage.model_dump(),
            "record_count": record_count,
        }
        meta_path = self._get_metadata_path(group_name, version)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

    def _load_metadata(self, group_name: str, version: int) -> Optional[Dict[str, Any]]:
        """Load feature group metadata from JSON."""
        meta_path = self._get_metadata_path(group_name, version)
        if meta_path.exists():
            with open(meta_path, "r") as f:
                return json.load(f)
        return None

    def insert_features(
        self,
        feature_group_name: str,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        version: int = 1,
    ) -> bool:
        """Insert features into local Parquet storage.

        Args:
            feature_group_name: Name of the feature group.
            df: DataFrame with feature data.
            metadata: Dataset metadata.
            version: Feature group version.

        Returns:
            True if insert succeeded.
        """
        # Validate synthetic data restrictions
        # Determine target type from group name
        target_type = "prod" if "_prod" in feature_group_name else "test"
        self._validate_insert(metadata, target_type)

        try:
            parquet_path = self._get_parquet_path(feature_group_name, version)

            # Save to Parquet
            df.to_parquet(parquet_path, index=False, engine="pyarrow")

            # Build and save lineage
            lineage = self._build_lineage(metadata, "1.0.0", "1.0")
            self._save_metadata(feature_group_name, version, metadata, lineage, len(df))

            logger.info(
                "Inserted %d records into %s (v%d)",
                len(df),
                feature_group_name,
                version,
            )
            return True

        except Exception as e:
            logger.error("Failed to insert into %s: %s", feature_group_name, str(e))
            return False

    def insert_targets(
        self,
        target_group_name: str,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        version: int = 1,
    ) -> bool:
        """Insert targets into local Parquet storage.

        Args:
            target_group_name: Name of the target group.
            df: DataFrame with target data.
            metadata: Dataset metadata.
            version: Feature group version.

        Returns:
            True if insert succeeded.
        """
        # Validate synthetic data restrictions
        target_type = "prod" if "_prod" in target_group_name else "test"
        self._validate_insert(metadata, target_type)

        try:
            parquet_path = self._get_parquet_path(target_group_name, version)
            df.to_parquet(parquet_path, index=False, engine="pyarrow")

            lineage = self._build_lineage(metadata, "1.0.0", "1.0")
            self._save_metadata(target_group_name, version, metadata, lineage, len(df))

            logger.info(
                "Inserted %d targets into %s (v%d)",
                len(df),
                target_group_name,
                version,
            )
            return True

        except Exception as e:
            logger.error("Failed to insert targets into %s: %s", target_group_name, str(e))
            return False

    def get_features(
        self,
        feature_group_name: str,
        version: int = 1,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Retrieve features from Parquet storage.

        Args:
            feature_group_name: Name of the feature group.
            version: Feature group version.
            columns: Specific columns to retrieve.

        Returns:
            DataFrame with features.
        """
        parquet_path = self._get_parquet_path(feature_group_name, version)

        if not parquet_path.exists():
            logger.warning("Feature group %s v%d not found", feature_group_name, version)
            return pd.DataFrame()

        try:
            if columns:
                df = pd.read_parquet(parquet_path, columns=columns, engine="pyarrow")
            else:
                df = pd.read_parquet(parquet_path, engine="pyarrow")

            logger.debug(
                "Retrieved %d records from %s (v%d)",
                len(df),
                feature_group_name,
                version,
            )
            return df

        except Exception as e:
            logger.error("Failed to retrieve from %s: %s", feature_group_name, str(e))
            return pd.DataFrame()

    def get_targets(
        self,
        target_group_name: str,
        version: int = 1,
    ) -> pd.DataFrame:
        """Retrieve targets from Parquet storage.

        Args:
            target_group_name: Name of the target group.
            version: Feature group version.

        Returns:
            DataFrame with targets.
        """
        return self.get_features(target_group_name, version)

    def delete_feature_group(
        self,
        group_name: str,
        version: int = 1,
    ) -> bool:
        """Delete a feature group.

        Args:
            group_name: Name of the feature group.
            version: Feature group version.

        Returns:
            True if deletion succeeded.
        """
        try:
            parquet_path = self._get_parquet_path(group_name, version)
            meta_path = self._get_metadata_path(group_name, version)

            if parquet_path.exists():
                parquet_path.unlink()
            if meta_path.exists():
                meta_path.unlink()

            logger.info("Deleted feature group %s v%d", group_name, version)
            return True

        except Exception as e:
            logger.error("Failed to delete %s: %s", group_name, str(e))
            return False

    def get_metadata(self, group_name: str, version: int = 1) -> Optional[Dict[str, Any]]:
        """Get metadata for a feature group.

        Args:
            group_name: Name of the feature group.
            version: Feature group version.

        Returns:
            Metadata dictionary or None.
        """
        return self._load_metadata(group_name, version)

    def list_feature_groups(self) -> List[str]:
        """List all available feature groups.

        Returns:
            List of feature group names.
        """
        if not self.base_path.exists():
            return []

        groups = []
        for item in self.base_path.iterdir():
            if item.is_dir() and item.name != "metadata":
                groups.append(item.name)

        return sorted(groups)

    def query_features(
        self,
        feature_group_name: str,
        sql_where: str,
        version: int = 1,
    ) -> pd.DataFrame:
        """Query features using DuckDB SQL.

        Args:
            feature_group_name: Name of the feature group.
            sql_where: SQL WHERE clause (without 'WHERE' keyword).
            version: Feature group version.

        Returns:
            Query results as DataFrame.
        """
        parquet_path = self._get_parquet_path(feature_group_name, version)

        if not parquet_path.exists():
            return pd.DataFrame()

        try:
            query = f"SELECT * FROM read_parquet('{parquet_path}') WHERE {sql_where}"
            result = self._connection.execute(query).fetchdf()
            return result

        except Exception as e:
            logger.error("DuckDB query failed: %s", str(e))
            return pd.DataFrame()

    def close(self) -> None:
        """Close DuckDB connection."""
        if self._connection:
            self._connection.close()
            logger.debug("Local feature store connection closed")
