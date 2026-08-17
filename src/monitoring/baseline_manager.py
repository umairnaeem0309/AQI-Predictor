"""
Baseline Management Module

Manages monitoring baselines with synthetic data rejection.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class BaselineMetadata:
    """Baseline metadata."""
    baseline_id: str
    baseline_type: str  # training, rolling, city_specific, seasonal
    dataset_type: str
    created_at: str
    data_start_date: str
    data_end_date: str
    feature_stats: Dict[str, Any]
    sample_count: int
    version: str
    feature_version: str
    model_version: str


class BaselineManager:
    """
    Baseline management for monitoring.
    
    Features:
    - Multiple baseline types (training, rolling, city-specific)
    - Synthetic data rejection for monitoring
    - Baseline versioning
    - Feature statistics storage
    """
    
    # Blocked dataset types for monitoring baselines
    BLOCKED_DATASET_TYPES = {"synthetic_test_data"}
    
    def __init__(self, baseline_dir: Path):
        """
        Initialize baseline manager.
        
        Args:
            baseline_dir: Directory to store baselines
        """
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
    
    def create_baseline(
        self,
        data: pd.DataFrame,
        baseline_type: str,
        dataset_type: str,
        feature_version: str = "1.0.0",
        model_version: str = "unknown",
        baseline_id: Optional[str] = None,
    ) -> BaselineMetadata:
        """
        Create a new baseline.
        
        Args:
            data: Reference data
            baseline_type: Type of baseline
            dataset_type: Type of dataset
            feature_version: Feature version
            model_version: Model version
            baseline_id: Custom baseline ID
            
        Returns:
            BaselineMetadata
            
        Raises:
            ValueError: If dataset_type is blocked for monitoring
        """
        # Reject synthetic data for monitoring baselines
        if dataset_type in self.BLOCKED_DATASET_TYPES:
            raise ValueError(
                f"Cannot create monitoring baseline with dataset_type='{dataset_type}'. "
                f"Blocked types: {self.BLOCKED_DATASET_TYPES}. "
                f"Use real_api_data for monitoring baselines."
            )
        
        # Generate baseline ID
        if baseline_id is None:
            baseline_id = f"baseline_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate feature statistics
        feature_stats = self._calculate_feature_stats(data)
        
        # Create metadata
        metadata = BaselineMetadata(
            baseline_id=baseline_id,
            baseline_type=baseline_type,
            dataset_type=dataset_type,
            created_at=datetime.now(timezone.utc).isoformat(),
            data_start_date=str(data["timestamp"].min()) if "timestamp" in data.columns else "",
            data_end_date=str(data["timestamp"].max()) if "timestamp" in data.columns else "",
            feature_stats=feature_stats,
            sample_count=len(data),
            version="1.0",
            feature_version=feature_version,
            model_version=model_version,
        )
        
        # Save baseline
        self._save_baseline(metadata, data)
        
        return metadata
    
    def load_baseline(
        self,
        baseline_id: str,
        reject_synthetic: bool = True,
    ) -> tuple[BaselineMetadata, pd.DataFrame]:
        """
        Load a baseline.
        
        Args:
            baseline_id: Baseline ID to load
            reject_synthetic: Reject synthetic data baselines
            
        Returns:
            Tuple of (BaselineMetadata, DataFrame)
            
        Raises:
            ValueError: If baseline not found or synthetic data rejected
        """
        baseline_path = self.baseline_dir / baseline_id
        
        if not baseline_path.exists():
            raise ValueError(f"Baseline not found: {baseline_id}")
        
        # Load metadata
        metadata_path = baseline_path / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata_dict = json.load(f)
        
        metadata = BaselineMetadata(**metadata_dict)
        
        # Check synthetic data
        if reject_synthetic and metadata.dataset_type in self.BLOCKED_DATASET_TYPES:
            raise ValueError(
                f"Cannot load synthetic baseline for monitoring: {baseline_id}. "
                f"dataset_type='{metadata.dataset_type}' is blocked."
            )
        
        # Load data
        data_path = baseline_path / "data.parquet"
        if data_path.exists():
            data = pd.read_parquet(data_path)
        else:
            data = pd.DataFrame()
        
        return metadata, data
    
    def list_baselines(
        self,
        baseline_type: Optional[str] = None,
        include_synthetic: bool = False,
    ) -> List[BaselineMetadata]:
        """
        List available baselines.
        
        Args:
            baseline_type: Filter by baseline type
            include_synthetic: Include synthetic data baselines
            
        Returns:
            List of BaselineMetadata
        """
        baselines = []
        
        for baseline_dir in self.baseline_dir.iterdir():
            if not baseline_dir.is_dir():
                continue
            
            metadata_path = baseline_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            
            with open(metadata_path, "r") as f:
                metadata_dict = json.load(f)
            
            metadata = BaselineMetadata(**metadata_dict)
            
            # Apply filters
            if baseline_type and metadata.baseline_type != baseline_type:
                continue
            if not include_synthetic and metadata.dataset_type in self.BLOCKED_DATASET_TYPES:
                continue
            
            baselines.append(metadata)
        
        return baselines
    
    def delete_baseline(self, baseline_id: str) -> bool:
        """Delete a baseline."""
        baseline_path = self.baseline_dir / baseline_id
        
        if not baseline_path.exists():
            return False
        
        # Delete all files in baseline directory
        for file in baseline_path.iterdir():
            file.unlink()
        
        baseline_path.rmdir()
        return True
    
    def _calculate_feature_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate feature statistics for baseline."""
        stats = {}
        
        for col in data.select_dtypes(include=["number"]).columns:
            stats[col] = {
                "mean": float(data[col].mean()),
                "std": float(data[col].std()),
                "min": float(data[col].min()),
                "max": float(data[col].max()),
                "percentiles": {
                    "25": float(data[col].quantile(0.25)),
                    "50": float(data[col].quantile(0.50)),
                    "75": float(data[col].quantile(0.75)),
                },
            }
        
        return stats
    
    def _save_baseline(self, metadata: BaselineMetadata, data: pd.DataFrame):
        """Save baseline to disk."""
        baseline_path = self.baseline_dir / metadata.baseline_id
        baseline_path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata_path = baseline_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(asdict(metadata), f, indent=2, default=str)
        
        # Save data
        if not data.empty:
            data_path = baseline_path / "data.parquet"
            data.to_parquet(data_path, index=False)
