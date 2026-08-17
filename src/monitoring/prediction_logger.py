"""
Prediction Logging Module

Logs predictions with security checks.
Storage: JSONL (current), database/object storage (future).
"""

import json
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class PredictionLog:
    """Single prediction log entry."""
    prediction_id: str
    timestamp: str
    city: str
    model_version: str
    input_features_hash: str  # Hash of features for security
    predictions: Dict[str, float]
    actual_values: Optional[Dict[str, float]]
    latency_ms: float
    confidence: Optional[float]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class PredictionLogger:
    """
    Prediction logging with security checks.
    
    Features:
    - JSONL storage format
    - Security: No secrets, API keys, or PII in logs
    - Feature hashing for privacy
    - Feedback loop support
    
    Storage Decision:
    - Initial: JSONL files (simple, portable)
    - Future: Database/object storage migration
    """
    
    # Sensitive fields that should never be logged
    SENSITIVE_FIELDS = {
        "api_key", "apikey", "api_secret", "secret",
        "password", "token", "auth", "credential",
        "email", "phone", "ssn", "credit_card",
        "user_id", "session_id", "ip_address",
    }
    
    def __init__(
        self,
        log_dir: Path,
        enable_security_checks: bool = True,
    ):
        """
        Initialize prediction logger.
        
        Args:
            log_dir: Directory to store prediction logs
            enable_security_checks: Enable security validation
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.enable_security_checks = enable_security_checks
    
    def log_prediction(
        self,
        city: str,
        model_version: str,
        input_features: Dict[str, Any],
        predictions: Dict[str, float],
        latency_ms: float,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PredictionLog:
        """
        Log a prediction.
        
        Args:
            city: City name
            model_version: Model version
            input_features: Input features (will be hashed)
            predictions: Prediction values
            latency_ms: Inference latency
            confidence: Prediction confidence
            metadata: Additional metadata
            
        Returns:
            PredictionLog entry
        """
        # Security check
        if self.enable_security_checks:
            self._validate_no_sensitive_data(input_features)
            if metadata:
                self._validate_no_sensitive_data(metadata)
        
        # Hash features for privacy
        features_hash = self._hash_features(input_features)
        
        # Create log entry
        log_entry = PredictionLog(
            prediction_id=f"pred_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            city=city,
            model_version=model_version,
            input_features_hash=features_hash,
            predictions=predictions,
            actual_values=None,
            latency_ms=latency_ms,
            confidence=confidence,
            metadata=metadata or {},
        )
        
        # Write to JSONL
        self._write_log(log_entry)
        
        return log_entry
    
    def update_actual_values(
        self,
        prediction_id: str,
        actual_values: Dict[str, float],
    ) -> bool:
        """
        Update log entry with actual values.
        
        Args:
            prediction_id: ID of prediction to update
            actual_values: Actual observed values
            
        Returns:
            True if updated, False if not found
        """
        # Find and update log file
        for log_file in self.log_dir.glob("**/*.jsonl"):
            lines = log_file.read_text().splitlines()
            updated = False
            
            for i, line in enumerate(lines):
                if not line.strip():
                    continue
                    
                entry = json.loads(line)
                if entry.get("prediction_id") == prediction_id:
                    entry["actual_values"] = actual_values
                    lines[i] = json.dumps(entry)
                    updated = True
                    break
            
            if updated:
                log_file.write_text("\n".join(lines) + "\n")
                return True
        
        return False
    
    def get_logs(
        self,
        city: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[PredictionLog]:
        """
        Retrieve prediction logs with optional filtering.
        
        Args:
            city: Filter by city
            start_time: Filter by start time
            end_time: Filter by end time
            
        Returns:
            List of PredictionLog entries
        """
        logs = []
        
        for log_file in self.log_dir.glob("**/*.jsonl"):
            for line in log_file.read_text().splitlines():
                if not line.strip():
                    continue
                    
                entry = json.loads(line)
                log = PredictionLog(**entry)
                
                # Apply filters
                if city and log.city != city:
                    continue
                if start_time:
                    log_time = datetime.fromisoformat(log.timestamp)
                    if log_time < start_time:
                        continue
                if end_time:
                    log_time = datetime.fromisoformat(log.timestamp)
                    if log_time > end_time:
                        continue
                
                logs.append(log)
        
        return logs
    
    def calculate_error_metrics(
        self,
        logs: List[PredictionLog],
    ) -> Dict[str, float]:
        """
        Calculate error metrics from logged predictions.
        
        Args:
            logs: List of PredictionLog entries with actual values
            
        Returns:
            Dictionary of error metrics
        """
        # Filter logs with actual values
        valid_logs = [l for l in logs if l.actual_values is not None]
        
        if not valid_logs:
            return {}
        
        # Calculate metrics for each horizon
        metrics = {}
        for horizon in ["24h", "48h", "72h"]:
            pred_key = f"aqi_{horizon}"
            actual_key = f"aqi_{horizon}"
            
            predictions = []
            actuals = []
            
            for log in valid_logs:
                if pred_key in log.predictions and actual_key in log.actual_values:
                    predictions.append(log.predictions[pred_key])
                    actuals.append(log.actual_values[actual_key])
            
            if predictions:
                import numpy as np
                predictions = np.array(predictions)
                actuals = np.array(actuals)
                
                mae = float(np.mean(np.abs(predictions - actuals)))
                rmse = float(np.sqrt(np.mean((predictions - actuals) ** 2)))
                
                metrics[f"mae_{horizon}"] = mae
                metrics[f"rmse_{horizon}"] = rmse
        
        return metrics
    
    def _validate_no_sensitive_data(self, data: Dict[str, Any]):
        """
        Validate that no sensitive data is present.
        
        Raises:
            ValueError: If sensitive data detected
        """
        for key, value in data.items():
            key_lower = key.lower()
            
            # Check for sensitive field names
            for sensitive in self.SENSITIVE_FIELDS:
                if sensitive in key_lower:
                    raise ValueError(
                        f"Sensitive field detected in prediction log: {key}. "
                        f"Remove or mask this field before logging."
                    )
            
            # Check for API key patterns in string values
            if isinstance(value, str):
                if len(value) > 20 and any(c.isalpha() for c in value):
                    # Potential API key pattern
                    if "key" in key_lower or "token" in key_lower or "secret" in key_lower:
                        raise ValueError(
                            f"Potential sensitive value detected in field: {key}. "
                            f"Mask or hash this value before logging."
                        )
    
    def _hash_features(self, features: Dict[str, Any]) -> str:
        """
        Hash features for privacy.
        
        Args:
            features: Input features
            
        Returns:
            SHA-256 hash of features
        """
        # Convert to JSON string and hash
        features_str = json.dumps(features, sort_keys=True, default=str)
        return hashlib.sha256(features_str.encode()).hexdigest()
    
    def _write_log(self, log_entry: PredictionLog):
        """Write log entry to JSONL file."""
        # Organize by date and city
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        city_dir = self.log_dir / date_str
        city_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = city_dir / f"{log_entry.city}_predictions.jsonl"
        
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry.to_dict()) + "\n")
