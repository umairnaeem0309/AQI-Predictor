"""
Drift Detection Module

Provides data drift detection using Evidently AI.
Supports PSI, KS test, and other drift metrics.

Evidently version: 0.7.21
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from evidently import Report
    from evidently.presets import DataDriftPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False


@dataclass
class DriftResult:
    """Result of drift detection analysis."""
    column_name: str
    drift_detected: bool
    drift_score: float
    drift_method: str
    threshold: float
    details: Dict[str, Any]
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class DriftReport:
    """Comprehensive drift report for a dataset."""
    report_id: str
    dataset_type: str
    baseline_version: str
    feature_version: str
    model_version: str
    reference_data_rows: int
    current_data_rows: int
    drift_results: List[DriftResult]
    overall_drift_detected: bool
    drift_summary: Dict[str, Any]
    generated_at: str = ""

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()


class DriftDetector:
    """
    Drift detection using Evidently AI.
    
    Supports:
    - Population Stability Index (PSI)
    - Kolmogorov-Smirnov test
    - Wasserstein distance
    - Chi-square test for categorical features
    """
    
    # Default thresholds
    DEFAULT_PSI_THRESHOLD = 0.1
    DEFAULT_KS_THRESHOLD = 0.05
    
    def __init__(
        self,
        psi_threshold: float = DEFAULT_PSI_THRESHOLD,
        ks_threshold: float = DEFAULT_KS_THRESHOLD,
    ):
        """
        Initialize drift detector.
        
        Args:
            psi_threshold: PSI threshold for drift detection (default: 0.1)
            ks_threshold: KS test p-value threshold (default: 0.05)
        """
        if not EVIDENTLY_AVAILABLE:
            raise ImportError(
                "Evidently is required for drift detection. "
                "Install with: pip install evidently>=0.7.0,<0.8.0"
            )
        
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
    
    def detect_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        dataset_type: str = "unknown",
        baseline_version: str = "unknown",
        feature_version: str = "unknown",
        model_version: str = "unknown",
    ) -> DriftReport:
        """
        Detect drift between reference and current datasets.
        
        Args:
            reference_data: Reference/baseline dataset
            current_data: Current dataset to check for drift
            dataset_type: Type of dataset (real_api_data, synthetic_test_data)
            baseline_version: Version of baseline data
            feature_version: Version of features
            model_version: Version of model
            
        Returns:
            DriftReport with drift detection results
        """
        # Validate inputs
        if reference_data.empty or current_data.empty:
            raise ValueError("Reference and current data cannot be empty")
        
        # Run Evidently drift detection
        report = Report([
            DataDriftPreset(method="psi"),
        ])
        
        evaluation = report.run(current_data, reference_data)
        report_dict = evaluation.dict()
        
        # Parse results
        drift_results = []
        overall_drift = False
        
        # Extract column-level results from Evidently v0.7+ format
        metrics = report_dict.get("metrics", [])
        for metric in metrics:
            metric_name = metric.get("metric_name", "")
            
            # Overall drift count metric
            if "DriftedColumnsCount" in metric_name:
                value = metric.get("value", {})
                if isinstance(value, dict):
                    drift_count = value.get("count", 0)
                    overall_drift = drift_count > 0
            
            # Per-column drift metric
            if metric_name.startswith("ValueDrift"):
                # Extract column name from config
                config = metric.get("config", {})
                col_name = config.get("column", "unknown")
                method = config.get("method", "psi")
                threshold_val = config.get("threshold", self.psi_threshold)
                
                # Value is the drift score; True/1.0 means drift detected
                score = metric.get("value", 0.0)
                if isinstance(score, bool):
                    detected = score
                    score_val = 1.0 if score else 0.0
                elif isinstance(score, (int, float)):
                    detected = score > threshold_val
                    score_val = float(score)
                else:
                    detected = False
                    score_val = 0.0
                
                drift_result = DriftResult(
                    column_name=col_name,
                    drift_detected=detected,
                    drift_score=score_val,
                    drift_method=method,
                    threshold=threshold_val,
                    details={
                        "metric_name": metric_name,
                    },
                )
                drift_results.append(drift_result)
        
        # Create summary
        drifted_columns = [r for r in drift_results if r.drift_detected]
        drift_summary = {
            "total_columns": len(drift_results),
            "drifted_columns": len(drifted_columns),
            "drift_percentage": len(drifted_columns) / len(drift_results) * 100 if drift_results else 0,
            "drifted_column_names": [r.column_name for r in drifted_columns],
        }
        
        # Create report
        drift_report = DriftReport(
            report_id=f"drift_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            dataset_type=dataset_type,
            baseline_version=baseline_version,
            feature_version=feature_version,
            model_version=model_version,
            reference_data_rows=len(reference_data),
            current_data_rows=len(current_data),
            drift_results=drift_results,
            overall_drift_detected=overall_drift,
            drift_summary=drift_summary,
        )
        
        return drift_report
    
    def save_report(
        self,
        report: DriftReport,
        output_dir: Path,
        save_html: bool = True,
    ) -> Path:
        """
        Save drift report to file.
        
        Args:
            report: DriftReport to save
            output_dir: Directory to save report
            save_html: Whether to save HTML version
            
        Returns:
            Path to saved report
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        json_path = output_dir / f"{report.report_id}.json"
        with open(json_path, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        return json_path
    
    def load_report(self, report_path: Path) -> DriftReport:
        """
        Load drift report from file.
        
        Args:
            report_path: Path to JSON report
            
        Returns:
            DriftReport object
        """
        with open(report_path, "r") as f:
            data = json.load(f)
        
        # Reconstruct DriftResult objects
        drift_results = [
            DriftResult(**result) for result in data.get("drift_results", [])
        ]
        
        return DriftReport(
            report_id=data["report_id"],
            dataset_type=data["dataset_type"],
            baseline_version=data["baseline_version"],
            feature_version=data["feature_version"],
            model_version=data["model_version"],
            reference_data_rows=data["reference_data_rows"],
            current_data_rows=data["current_data_rows"],
            drift_results=drift_results,
            overall_drift_detected=data["overall_drift_detected"],
            drift_summary=data["drift_summary"],
            generated_at=data.get("generated_at", ""),
        )
