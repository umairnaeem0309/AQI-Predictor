"""
Dataset Versioning

Manages dataset versions with metadata and lineage.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class DatasetVersion:
    """Dataset version metadata."""

    dataset_id: str
    dataset_type: str  # real_api_data, synthetic_test_data
    source: str
    date_range_start: str
    date_range_end: str
    cities: List[str]
    resolution: str
    total_observations: int
    features: int
    quality_score: float
    created_at: str
    approved_for_training: bool
    approved_for_evaluation: bool
    version: str
    parent_version: Optional[str] = None
    notes: Optional[str] = None


class DatasetVersionManager:
    """
    Manages dataset versions.

    Features:
    - Version tracking
    - Metadata storage
    - Lineage tracking
    - Approval workflow
    """

    def __init__(self, version_dir: Optional[Path] = None):
        """
        Initialize dataset version manager.

        Args:
            version_dir: Directory for version metadata
        """
        self.version_dir = version_dir or Path("data/processed/versions")
        self.version_dir.mkdir(parents=True, exist_ok=True)

    def create_version(
        self,
        dataset_type: str,
        source: str,
        date_range_start: str,
        date_range_end: str,
        cities: List[str],
        resolution: str,
        total_observations: int,
        features: int,
        quality_score: float,
        version: Optional[str] = None,
        parent_version: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> DatasetVersion:
        """
        Create a new dataset version.

        Args:
            dataset_type: Type of dataset
            source: Data source
            date_range_start: Start date
            date_range_end: End date
            cities: List of cities
            resolution: Data resolution
            total_observations: Total observations
            features: Number of features
            quality_score: Quality score (0-1)
            version: Version string (auto-generated if None)
            parent_version: Parent version ID
            notes: Additional notes

        Returns:
            DatasetVersion object
        """
        # Generate version ID
        if version is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
            version = f"{dataset_type}_{timestamp}_v1.0"

        # Create version
        dataset_version = DatasetVersion(
            dataset_id=version,
            dataset_type=dataset_type,
            source=source,
            date_range_start=date_range_start,
            date_range_end=date_range_end,
            cities=cities,
            resolution=resolution,
            total_observations=total_observations,
            features=features,
            quality_score=quality_score,
            created_at=datetime.now(timezone.utc).isoformat(),
            approved_for_training=dataset_type == "real_api_data",
            approved_for_evaluation=dataset_type == "real_api_data",
            version=version,
            parent_version=parent_version,
            notes=notes,
        )

        # Save version metadata
        self._save_version(dataset_version)

        return dataset_version

    def get_version(self, version_id: str) -> Optional[DatasetVersion]:
        """
        Get a dataset version.

        Args:
            version_id: Version ID

        Returns:
            DatasetVersion if found, None otherwise
        """
        version_file = self.version_dir / f"{version_id}.json"

        if not version_file.exists():
            return None

        with open(version_file, "r") as f:
            data = json.load(f)

        return DatasetVersion(**data)

    def list_versions(
        self,
        dataset_type: Optional[str] = None,
    ) -> List[DatasetVersion]:
        """
        List all versions.

        Args:
            dataset_type: Filter by dataset type

        Returns:
            List of DatasetVersion objects
        """
        versions = []

        for version_file in self.version_dir.glob("*.json"):
            with open(version_file, "r") as f:
                data = json.load(f)

            version = DatasetVersion(**data)

            if dataset_type and version.dataset_type != dataset_type:
                continue

            versions.append(version)

        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def approve_version(
        self,
        version_id: str,
        approved_for_training: bool = True,
        approved_for_evaluation: bool = True,
    ) -> bool:
        """
        Approve a dataset version.

        Args:
            version_id: Version ID
            approved_for_training: Approve for training
            approved_for_evaluation: Approve for evaluation

        Returns:
            True if approved, False if not found
        """
        version = self.get_version(version_id)
        if version is None:
            return False

        version.approved_for_training = approved_for_training
        version.approved_for_evaluation = approved_for_evaluation

        self._save_version(version)
        return True

    def _save_version(self, version: DatasetVersion):
        """Save version metadata."""
        version_file = self.version_dir / f"{version.dataset_id}.json"

        with open(version_file, "w") as f:
            json.dump(asdict(version), f, indent=2)

    def get_latest_approved(self) -> Optional[DatasetVersion]:
        """
        Get latest approved version for training.

        Returns:
            Latest approved DatasetVersion or None
        """
        versions = self.list_versions(dataset_type="real_api_data")

        for version in versions:
            if version.approved_for_training:
                return version

        return None
