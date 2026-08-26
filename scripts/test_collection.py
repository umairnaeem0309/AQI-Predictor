#!/usr/bin/env python3
"""Test real data collection with fixed AQICN station IDs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import load_environment
load_environment()

from src.data.real_data_collector import RealDataCollector

collector = RealDataCollector()
df = collector.collect_round(save_raw=True)

print(f"Collection result: {len(df)} observations")
if not df.empty:
    cols_of_interest = [
        "location_id", "aqi", "raw_response_time", "collected_at",
        "is_training_valid", "staleness_reason", "data_source",
    ]
    available = [c for c in cols_of_interest if c in df.columns]
    print(df[available].to_string())
