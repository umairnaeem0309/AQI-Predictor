#!/usr/bin/env python3
"""Report collection health for GitHub Actions step summary."""

import json
from pathlib import Path


def main():
    log_path = Path("data/collection_health.json")
    if not log_path.exists():
        print("No collection log available")
        return

    with open(log_path) as f:
        log = json.load(f)

    if not log:
        print("Collection log is empty")
        return

    latest = log[-1]
    print(f"**Cities attempted:** {latest.get('cities_attempted', 'N/A')}")
    print(f"**Observations:** {latest.get('observations_collected', 'N/A')}")
    print(f"**Training valid:** {latest.get('training_valid_observations', 'N/A')}")
    persisted = latest.get("hopsworks_persisted", False)
    print(f"**Hopsworks:** {'PASS' if persisted else 'FAIL'}")
    print(f"**Status:** {latest.get('status', 'N/A')}")


if __name__ == "__main__":
    main()
