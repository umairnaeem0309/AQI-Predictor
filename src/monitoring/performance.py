"""
Performance Monitoring Module

Tracks model performance metrics over time.
Provides rolling windows and degradation detection.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetric:
    """Single performance metric measurement."""

    metric_name: str
    value: float
    window: str  # 24h, 7d, 30d
    horizon: str  # 24h, 48h, 72h
    city: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""

    report_id: str
    dataset_type: str
    baseline_version: str
    feature_version: str
    model_version: str
    metrics: List[PerformanceMetric]
    degradation_detected: bool
    degradation_summary: Dict[str, Any]
    generated_at: str = ""

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()


class PerformanceMonitor:
    """
    Model performance monitoring.

    Tracks:
    - MAE, RMSE, R² over rolling windows
    - Performance degradation detection
    - City and horizon-specific metrics
    """

    # Degradation thresholds
    DEFAULT_MAE_INCREASE_THRESHOLD = 0.2  # 20% increase
    DEFAULT_RMSE_INCREASE_THRESHOLD = 0.2
    DEFAULT_R2_DECREASE_THRESHOLD = 0.1  # 10% decrease

    def __init__(
        self,
        mae_threshold: float = DEFAULT_MAE_INCREASE_THRESHOLD,
        rmse_threshold: float = DEFAULT_RMSE_INCREASE_THRESHOLD,
        r2_threshold: float = DEFAULT_R2_DECREASE_THRESHOLD,
    ):
        """
        Initialize performance monitor.

        Args:
            mae_threshold: MAE increase threshold for degradation (default: 0.2)
            rmse_threshold: RMSE increase threshold for degradation (default: 0.2)
            r2_threshold: R² decrease threshold for degradation (default: 0.1)
        """
        self.mae_threshold = mae_threshold
        self.rmse_threshold = rmse_threshold
        self.r2_threshold = r2_threshold

    def calculate_rolling_metrics(
        self,
        predictions_df: pd.DataFrame,
        window: str = "24h",
    ) -> Dict[str, float]:
        """
        Calculate rolling performance metrics.

        Args:
            predictions_df: DataFrame with predictions and actuals
            window: Rolling window size (24h, 7d, 30d)

        Returns:
            Dictionary of metrics
        """
        if predictions_df.empty:
            return {"mae": 0.0, "rmse": 0.0, "r2": 0.0}

        # Ensure timestamp column is datetime
        if "timestamp" in predictions_df.columns:
            df = predictions_df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")

            # Apply rolling window
            window_hours = self._parse_window(window)
            cutoff_time = df["timestamp"].max() - pd.Timedelta(hours=window_hours)
            df = df[df["timestamp"] >= cutoff_time]

        # Calculate metrics for each horizon
        metrics = {}
        for horizon in ["24h", "48h", "72h"]:
            pred_col = f"aqi_{horizon}"
            actual_col = f"actual_{horizon}"

            if pred_col in df.columns and actual_col in df.columns:
                valid_mask = df[pred_col].notna() & df[actual_col].notna()
                if valid_mask.sum() > 0:
                    predictions = df.loc[valid_mask, pred_col].values
                    actuals = df.loc[valid_mask, actual_col].values

                    mae = np.mean(np.abs(predictions - actuals))
                    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

                    # R² calculation
                    ss_res = np.sum((actuals - predictions) ** 2)
                    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

                    metrics[f"mae_{horizon}"] = float(mae)
                    metrics[f"rmse_{horizon}"] = float(rmse)
                    metrics[f"r2_{horizon}"] = float(r2)

        # Average across horizons
        if metrics:
            metrics["mae"] = np.mean([v for k, v in metrics.items() if k.startswith("mae_")])
            metrics["rmse"] = np.mean([v for k, v in metrics.items() if k.startswith("rmse_")])
            metrics["r2"] = np.mean([v for k, v in metrics.items() if k.startswith("r2_")])

        return metrics

    def detect_degradation(
        self,
        current_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Detect performance degradation.

        Args:
            current_metrics: Current performance metrics
            baseline_metrics: Baseline performance metrics

        Returns:
            Tuple of (degradation_detected, degradation_details)
        """
        degradation = False
        details = {}

        # Check MAE
        if "mae" in current_metrics and "mae" in baseline_metrics:
            mae_increase = (current_metrics["mae"] - baseline_metrics["mae"]) / baseline_metrics[
                "mae"
            ]
            details["mae_change"] = float(mae_increase)
            if mae_increase > self.mae_threshold:
                degradation = True
                details["mae_degraded"] = True

        # Check RMSE
        if "rmse" in current_metrics and "rmse" in baseline_metrics:
            rmse_increase = (current_metrics["rmse"] - baseline_metrics["rmse"]) / baseline_metrics[
                "rmse"
            ]
            details["rmse_change"] = float(rmse_increase)
            if rmse_increase > self.rmse_threshold:
                degradation = True
                details["rmse_degraded"] = True

        # Check R²
        if "r2" in current_metrics and "r2" in baseline_metrics:
            r2_decrease = (baseline_metrics["r2"] - current_metrics["r2"]) / baseline_metrics["r2"]
            details["r2_change"] = float(r2_decrease)
            if r2_decrease > self.r2_threshold:
                degradation = True
                details["r2_degraded"] = True

        return degradation, details

    def generate_report(
        self,
        predictions_df: pd.DataFrame,
        baseline_metrics: Dict[str, float],
        dataset_type: str = "unknown",
        baseline_version: str = "unknown",
        feature_version: str = "unknown",
        model_version: str = "unknown",
        city: str = "all",
    ) -> PerformanceReport:
        """
        Generate comprehensive performance report.

        Args:
            predictions_df: DataFrame with predictions and actuals
            baseline_metrics: Baseline performance metrics
            dataset_type: Type of dataset
            baseline_version: Version of baseline
            feature_version: Version of features
            model_version: Version of model
            city: City filter (or "all")

        Returns:
            PerformanceReport
        """
        metrics_list = []

        # Calculate metrics for different windows
        for window in ["24h", "7d", "30d"]:
            window_metrics = self.calculate_rolling_metrics(predictions_df, window)

            for horizon in ["24h", "48h", "72h"]:
                for metric_name in ["mae", "rmse", "r2"]:
                    key = f"{metric_name}_{horizon}"
                    if key in window_metrics:
                        metrics_list.append(
                            PerformanceMetric(
                                metric_name=metric_name,
                                value=window_metrics[key],
                                window=window,
                                horizon=horizon,
                                city=city,
                            )
                        )

        # Detect degradation
        current_metrics = self.calculate_rolling_metrics(predictions_df, "30d")
        degradation_detected, degradation_summary = self.detect_degradation(
            current_metrics, baseline_metrics
        )

        return PerformanceReport(
            report_id=f"perf_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            dataset_type=dataset_type,
            baseline_version=baseline_version,
            feature_version=feature_version,
            model_version=model_version,
            metrics=metrics_list,
            degradation_detected=degradation_detected,
            degradation_summary=degradation_summary,
        )

    def save_report(self, report: PerformanceReport, output_dir: Path) -> Path:
        """Save performance report to file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{report.report_id}.json"

        with open(json_path, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)

        return json_path

    def _parse_window(self, window: str) -> int:
        """Parse window string to hours."""
        if window == "24h":
            return 24
        elif window == "7d":
            return 168
        elif window == "30d":
            return 720
        else:
            return 24
