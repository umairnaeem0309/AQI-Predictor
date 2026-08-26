#!/usr/bin/env python3
"""
Quality Gate Script

Validates data quality and sufficiency before training.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class QualityGateError(Exception):
    """Quality gate error."""
    pass


class DataQualityGate:
    """Data quality gate for training readiness."""
    
    # Quality thresholds
    MIN_COMPLETENESS = 0.90  # 90% completeness required
    MAX_STALENESS_HOURS = 2  # Data must be within 2 hours
    MIN_OBSERVATIONS_PER_CITY = 500  # Minimum observations per city
    MIN_DAYS_COLLECTED = 21  # Minimum 21 days of data
    MAX_DUPLICATE_RATIO = 0.01  # Max 1% duplicates
    
    def __init__(self):
        self.results = {}
    
    def check_completeness(self, df: pd.DataFrame) -> dict:
        """
        Check data completeness.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Completeness check results
        """
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness = 1 - (missing_cells / total_cells)
        
        passed = completeness >= self.MIN_COMPLETENESS
        
        return {
            "check": "completeness",
            "passed": passed,
            "score": float(completeness),
            "threshold": self.MIN_COMPLETENESS,
            "missing_cells": int(missing_cells),
            "total_cells": int(total_cells),
        }
    
    def check_freshness(self, df: pd.DataFrame) -> dict:
        """
        Check data freshness.
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            Freshness check results
        """
        if "timestamp" not in df.columns:
            return {
                "check": "freshness",
                "passed": False,
                "error": "No timestamp column",
            }
        
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            latest_timestamp = df["timestamp"].max()
            now = pd.Timestamp.now(tz="UTC")
            
            # Ensure timezone aware
            if latest_timestamp.tzinfo is None:
                latest_timestamp = latest_timestamp.tz_localize("UTC")
            
            age_hours = (now - latest_timestamp).total_seconds() / 3600
            passed = age_hours <= self.MAX_STALENESS_HOURS
            
            return {
                "check": "freshness",
                "passed": passed,
                "age_hours": float(age_hours),
                "threshold_hours": self.MAX_STALENESS_HOURS,
                "latest_timestamp": latest_timestamp.isoformat(),
            }
        except Exception as e:
            return {
                "check": "freshness",
                "passed": False,
                "error": str(e),
            }
    
    def check_schema(self, df: pd.DataFrame, required_columns: List[str]) -> dict:
        """
        Check schema compliance.
        
        Args:
            df: DataFrame to check
            required_columns: Required column names
            
        Returns:
            Schema check results
        """
        missing_columns = set(required_columns) - set(df.columns)
        extra_columns = set(df.columns) - set(required_columns)
        
        passed = len(missing_columns) == 0
        
        return {
            "check": "schema",
            "passed": passed,
            "missing_columns": list(missing_columns),
            "extra_columns": list(extra_columns),
            "total_columns": len(df.columns),
        }
    
    def check_duplicates(self, df: pd.DataFrame) -> dict:
        """
        Check for duplicates.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Duplicate check results
        """
        total_rows = len(df)
        duplicate_rows = df.duplicated().sum()
        duplicate_ratio = duplicate_rows / total_rows if total_rows > 0 else 0
        
        passed = duplicate_ratio <= self.MAX_DUPLICATE_RATIO
        
        return {
            "check": "duplicates",
            "passed": passed,
            "duplicate_rows": int(duplicate_rows),
            "total_rows": int(total_rows),
            "duplicate_ratio": float(duplicate_ratio),
            "threshold": self.MAX_DUPLICATE_RATIO,
        }
    
    def check_data_sufficiency(self, df: pd.DataFrame) -> dict:
        """
        Check data sufficiency for training.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Sufficiency check results
        """
        checks = {}
        
        # Check total observations
        total_observations = len(df)
        checks["total_observations"] = {
            "value": total_observations,
            "threshold": self.MIN_OBSERVATIONS_PER_CITY * 3,  # 3 cities
            "passed": total_observations >= self.MIN_OBSERVATIONS_PER_CITY * 3,
        }
        
        # Check observations per city
        if "location_id" in df.columns:
            city_counts = df["location_id"].value_counts().to_dict()
            checks["per_city"] = {}
            for city, count in city_counts.items():
                checks["per_city"][city] = {
                    "value": count,
                    "threshold": self.MIN_OBSERVATIONS_PER_CITY,
                    "passed": count >= self.MIN_OBSERVATIONS_PER_CITY,
                }
        
        # Check date range
        if "timestamp" in df.columns:
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                date_range = (df["timestamp"].max() - df["timestamp"].min()).days
                checks["date_range_days"] = {
                    "value": date_range,
                    "threshold": self.MIN_DAYS_COLLECTED,
                    "passed": date_range >= self.MIN_DAYS_COLLECTED,
                }
            except Exception:
                checks["date_range_days"] = {
                    "passed": False,
                    "error": "Could not parse timestamps",
                }
        
        # Overall sufficiency
        all_passed = all(
            check.get("passed", False)
            for check in checks.values()
            if isinstance(check, dict)
        )
        all_passed = all_passed and all(
            city_check.get("passed", False)
            for city_check in checks.get("per_city", {}).values()
        )
        
        return {
            "check": "data_sufficiency",
            "passed": all_passed,
            "details": checks,
        }
    
    def run_all_checks(
        self,
        df: pd.DataFrame,
        required_columns: List[str] = None,
    ) -> dict:
        """
        Run all quality checks.
        
        Args:
            df: DataFrame to check
            required_columns: Required columns for schema check
            
        Returns:
            Combined quality gate results
        """
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {},
            "all_passed": True,
        }
        
        # Run checks
        results["checks"]["completeness"] = self.check_completeness(df)
        results["checks"]["freshness"] = self.check_freshness(df)
        results["checks"]["duplicates"] = self.check_duplicates(df)
        results["checks"]["data_sufficiency"] = self.check_data_sufficiency(df)
        
        if required_columns:
            results["checks"]["schema"] = self.check_schema(df, required_columns)
        
        # Check overall status
        for check_name, check_results in results["checks"].items():
            if not check_results.get("passed", False):
                results["all_passed"] = False
                break
        
        return results
    
    def save_results(self, results: dict, output_dir: Path = None):
        """Save quality gate results."""
        if output_dir is None:
            output_dir = project_root / "data" / "quality_reports"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"quality_{timestamp}.json"
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_file}")
        return output_file


def _ensure_utf8_stdout():
    """Ensure stdout supports Unicode symbols on Windows consoles."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def main():
    """Main quality gate entry point."""
    import argparse
    
    _ensure_utf8_stdout()
    parser = argparse.ArgumentParser(description="Run data quality gate")
    parser.add_argument("data_file", help="Path to CSV data file")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    
    args = parser.parse_args()
    
    # Load data
    try:
        df = pd.read_csv(args.data_file)
        print(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Run quality gate
    gate = DataQualityGate()
    results = gate.run_all_checks(df)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Data Quality Gate Results")
    print("=" * 60)
    
    for check_name, check_results in results["checks"].items():
        status = "✅ PASSED" if check_results.get("passed") else "❌ FAILED"
        print(f"{check_name}: {status}")
        
        if "score" in check_results:
            print(f"  Score: {check_results['score']:.2f}")
        if "error" in check_results:
            print(f"  Error: {check_results['error']}")
    
    print("\n" + "=" * 60)
    if results["all_passed"]:
        print("✅ All quality checks passed! Data is ready for training.")
    else:
        print("❌ Some quality checks failed! Data needs improvement.")
    print("=" * 60)
    
    # Save results
    if args.save:
        gate.save_results(results)
    
    sys.exit(0 if results["all_passed"] else 1)


if __name__ == "__main__":
    main()
