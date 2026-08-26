#!/usr/bin/env python3
"""Test updated quality gate with freshness validation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
from scripts.quality_gate import DataQualityGate

# Load the latest master observations
master_file = Path("data/raw/real/master_observations.csv")
if not master_file.exists():
    print("No master observations file found")
    sys.exit(1)

df = pd.read_csv(master_file)
print(f"Loaded {len(df)} observations")
print(f"Columns: {list(df.columns)}")

gate = DataQualityGate()
results = gate.run_all_checks(df)

print("\n" + "=" * 60)
print("Quality Gate Results")
print("=" * 60)
for check_name, check_results in results["checks"].items():
    status = "PASSED" if check_results.get("passed") else "FAILED"
    print(f"\n  {check_name}: {status}")
    for key, val in check_results.items():
        if key in ("check", "passed"):
            continue
        if isinstance(val, dict):
            for k2, v2 in val.items():
                print(f"    {key}.{k2}: {v2}")
        else:
            print(f"    {key}: {val}")

print(f"\nOverall: {'PASSED' if results['all_passed'] else 'FAILED'}")
