"""
Feature Store Interface — Abstract base class for all feature store implementations.

Defines the contract for feature storage operations:
- Insert features and targets
- Retrieve features and targets
- Delete feature groups
- Manage feature metadata and versions

Synthetic data safety:
- All insert operations validate dataset_type and training approval flags
- Datasets not approved for training cannot be inserted into production groups
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd

from src.feature_store.schemas import (
    FeatureGroupMetadata,
    DatasetMetadata,
    LineageMetadata,
    DatasetType,
)

logger = logging.getLogger(__name__)


class FeatureStoreInterface(ABC):
    """Abstract interface for feature storage.

    All feature store implementations must implement this interface.
    The interface enforces synthetic data safety by validating
    dataset metadata before insert operations.

    Implementations:
    - HopsworksStore: Cloud feature store via Hopsworks
    - LocalStore: Local DuckDB + Parquet storage
    """

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the feature store."""
        pass

    @abstractmethod
    def insert_features(
        self,
        feature_group_name: str,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        version: int = 1,
    ) -> bool:
        """Insert features into a feature group.

        Args:
            feature_group_name: Name of the feature group.
            df: DataFrame with feature data.
            metadata: Dataset metadata (validates synthetic data restrictions).
            version: Feature group version.

        Returns:
            True if insert succeeded, False otherwise.

        Raises:
            ValueError: If dataset is not approved for the target group type.
        """
        pass

    @abstractmethod
    def insert_targets(
        self,
        target_group_name: str,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        version: int = 1,
    ) -> bool:
        """Insert targets into a feature group.

        Args:
            target_group_name: Name of the target group.
            df: DataFrame with target data.
            metadata: Dataset metadata.
            version: Feature group version.

        Returns:
            True if insert succeeded, False otherwise.
        """
        pass

    @abstractmethod
    def get_features(
        self,
        feature_group_name: str,
        version: int = 1,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Retrieve features from a feature group.

        Args:
            feature_group_name: Name of the feature group.
            version: Feature group version.
            columns: Specific columns to retrieve (None for all).

        Returns:
            DataFrame with features.
        """
        pass

    @abstractmethod
    def get_targets(
        self,
        target_group_name: str,
        version: int = 1,
    ) -> pd.DataFrame:
        """Retrieve targets from a feature group.

        Args:
            target_group_name: Name of the target group.
            version: Feature group version.

        Returns:
            DataFrame with targets.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_metadata(self, group_name: str, version: int = 1) -> Optional[Dict[str, Any]]:
        """Get metadata for a feature group.

        Args:
            group_name: Name of the feature group.
            version: Feature group version.

        Returns:
            Metadata dictionary or None.
        """
        pass

    def _validate_insert(
        self,
        metadata: DatasetMetadata,
        target_group_type: str,
    ) -> None:
        """Validate dataset metadata before insert.

        Prevents:
        - Inserting synthetic data into production groups
        - Inserting data not approved for training into training groups

        Args:
            metadata: Dataset metadata to validate.
            target_group_type: Type of target group (test or prod).

        Raises:
            ValueError: If validation fails.
        """
        # Check: synthetic data cannot go into production groups
        if metadata.dataset_type == DatasetType.SYNTHETIC_TEST and target_group_type == "prod":
            raise ValueError(
                f"Cannot insert synthetic test data into production feature group. "
                f"Dataset type: {metadata.dataset_type.value}"
            )

        # Check: data must be approved for training
        if not metadata.approved_for_training and target_group_type == "prod":
            logger.warning(
                "Dataset %s is not approved for training — insert into prod blocked",
                metadata.dataset_version,
            )
            raise ValueError(
                f"Dataset {metadata.dataset_version} is not approved for training. "
                f"Cannot insert into production feature group."
            )

        logger.debug(
            "Insert validation passed: dataset=%s, type=%s, approved=%s, target=%s",
            metadata.dataset_version,
            metadata.dataset_type.value,
            metadata.approved_for_training,
            target_group_type,
        )

    def _build_lineage(
        self,
        metadata: DatasetMetadata,
        feature_version: str,
        schema_version: str,
    ) -> LineageMetadata:
        """Build lineage metadata for feature group insertion.

        Args:
            metadata: Dataset metadata.
            feature_version: Feature definition version.
            schema_version: Schema version.

        Returns:
            LineageMetadata with all required fields.
        """
        from datetime import datetime, timezone
        return LineageMetadata(
            feature_version=feature_version,
            schema_version=schema_version,
            source_dataset_version=metadata.dataset_version,
            creation_timestamp=datetime.now(timezone.utc).isoformat(),
            dataset_type=metadata.dataset_type.value,
        )
