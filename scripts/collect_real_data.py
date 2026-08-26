#!/usr/bin/env python3
"""
Real Data Collection Entrypoint

Collects one round of real data from OpenWeather and AQICN.
Designed for scheduled execution (Windows Task Scheduler / cron).

Responsibilities:
- Load environment safely
- Initialize collector
- Collect one round
- Persist observations
- Update usage/audit metrics
- Exit with appropriate status code

Usage:
    conda run -n aqi-predictor python scripts/collect_real_data.py
"""
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")
from src.config import load_environment
load_environment()

from src.data.real_data_collector import RealDataCollector

# Lock file to prevent overlapping runs
LOCK_FILE = project_root / "data" / "raw" / "real" / ".collection.lock"
LOG_DIR = project_root / "data" / "raw" / "real" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def acquire_lock():
    """Acquire collection lock to prevent overlapping runs."""
    if LOCK_FILE.exists():
        # Check if lock is stale (older than 10 minutes)
        lock_age = time.time() - LOCK_FILE.stat().st_mtime
        if lock_age > 600:
            print(f"WARNING: Stale lock file ({lock_age:.0f}s old), removing")
            LOCK_FILE.unlink()
        else:
            print(f"ERROR: Collection already running (lock age: {lock_age:.0f}s)")
            return False
    LOCK_FILE.write_text(str(os.getpid()))
    return True


def release_lock():
    """Release collection lock."""
    if LOCK_FILE.exists():
        LOCK_FILE.unlink()


def main():
    """Main collection entrypoint."""
    start_time = datetime.now(timezone.utc)

    # Acquire lock
    if not acquire_lock():
        sys.exit(1)

    try:
        # Initialize collector
        collector = RealDataCollector(
            output_dir=project_root / "data" / "raw" / "real"
        )

        # Collect one round
        df = collector.collect_round()

        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        # Log result
        log_entry = {
            "scheduled_time": start_time.isoformat(),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": round(duration, 2),
            "observations": len(df),
            "status": "success" if len(df) > 0 else "empty",
            "cities": df["location_id"].tolist() if len(df) > 0 else [],
        }

        log_file = LOG_DIR / f"collection_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, "w") as f:
            json.dump(log_entry, f, indent=2)

        print(f"\nCollection complete: {len(df)} observations in {duration:.1f}s")
        print(f"Log: {log_file}")

        if len(df) > 0:
            sys.exit(0)
        else:
            print("WARNING: No observations collected")
            sys.exit(1)

    except Exception as e:
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        # Log error
        log_entry = {
            "scheduled_time": start_time.isoformat(),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": round(duration, 2),
            "observations": 0,
            "status": "error",
            "error": str(e),
        }

        log_file = LOG_DIR / f"collection_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, "w") as f:
            json.dump(log_entry, f, indent=2)

        print(f"\nCollection FAILED: {e}")
        sys.exit(1)

    finally:
        release_lock()


if __name__ == "__main__":
    main()
