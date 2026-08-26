#!/usr/bin/env python3
"""Hopsworks full connection and feature store validation."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import load_environment
load_environment()

host = os.environ.get("HOPSWORKS_HOST")
api_key = os.environ.get("HOPSWORKS_API_KEY")
project_name = os.environ.get("HOPSWORKS_PROJECT", "aqi_predictor")

print("Hopsworks Connection Validation")
print("=" * 50)
print(f"  Host: configured ({len(host)} chars)")
print(f"  API Key: configured ({len(api_key)} chars)")
print(f"  Project: {project_name}")
print()

try:
    import hopsworks
    print(f"  hopsworks version: {hopsworks.__version__}")
except ImportError as e:
    print(f"  FAIL: hopsworks not importable: {e}")
    sys.exit(1)

try:
    project = hopsworks.login(
        host=host,
        api_key_value=api_key,
        project=project_name,
    )
    print("  Connection: SUCCESS")
except Exception as e:
    print(f"  Connection: FAILED - {e}")
    sys.exit(1)

# Try feature store access (v3.7+ API)
try:
    fs = project.get_feature_store()
    print(f"  Feature Store: accessible via get_feature_store()")
except AttributeError:
    try:
        fs = project.feature_store
        print(f"  Feature Store: accessible via .feature_store")
    except Exception as e:
        print(f"  Feature Store: FAILED - {e}")
        fs = None
except Exception as e:
    print(f"  Feature Store: FAILED - {e}")
    fs = None

if fs:
    try:
        fgs = fs.get_feature_groups()
        print(f"  Existing feature groups: {len(fgs)}")
        for fg in fgs:
            print(f"    - {fg.name} (v{fg.version})")
    except Exception as e:
        print(f"  Feature group listing: FAILED - {e}")

    try:
        fvs = fs.get_feature_views()
        print(f"  Existing feature views: {len(fvs)}")
    except Exception as e:
        print(f"  Feature view listing: FAILED - {e}")

print()
print("HOPSWORKS VALIDATION COMPLETE")
print("=" * 50)
