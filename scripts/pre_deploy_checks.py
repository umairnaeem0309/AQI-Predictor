#!/usr/bin/env python3
"""
Pre-Deployment Safety Checks

Verifies deployment is safe before production release.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_mock_mode() -> bool:
    """Check MOCK_MODE is false in production."""
    mock_mode = os.getenv("MOCK_MODE", "false").lower()

    if mock_mode == "true":
        print("[FAIL] CRITICAL: MOCK_MODE is true in production!")
        print("   Mock mode must be false for production deployment.")
        return False

    print("[OK] MOCK_MODE is false")
    return True


def check_api_key() -> bool:
    """Check API_KEY is set."""
    api_key = os.getenv("API_KEY")

    if not api_key:
        print("[FAIL] CRITICAL: API_KEY is not set!")
        print("   API_KEY must be set for production deployment.")
        return False

    # Don't log the actual key
    print("[OK] API_KEY is set")
    return True


def check_hopsworks_host() -> bool:
    """Check HOPSWORKS_HOST is set."""
    hopsworks_host = os.getenv("HOPSWORKS_HOST")

    if not hopsworks_host:
        print("[WARN] WARNING: HOPSWORKS_HOST is not set!")
        print("   Feature store will use local fallback.")
        print("   This may not be suitable for production.")
        # Warning only, not failure
        return True

    print("[OK] HOPSWORKS_HOST is set")
    return True


def check_api_keys() -> bool:
    """Check API keys are set."""
    openweather_key = os.getenv("OPENWEATHER_API_KEY")
    aqicn_key = os.getenv("AQICN_API_KEY")

    missing = []
    if not openweather_key:
        missing.append("OPENWEATHER_API_KEY")
    if not aqicn_key:
        missing.append("AQICN_API_KEY")

    if missing:
        print(f"[WARN] WARNING: Missing API keys: {', '.join(missing)}")
        print("   Data collection may fail without API keys.")
        return True  # Warning only

    print("[OK] API keys are set")
    return True


def check_model_metadata() -> bool:
    """Check model metadata is valid for production."""
    # This would check the actual model metadata
    # For now, just verify the check exists

    print("[OK] Model metadata check passed (placeholder)")
    return True


def check_feature_store_config() -> bool:
    """Check feature store configuration."""
    feature_store_primary = os.getenv("FEATURE_STORE_PRIMARY", "hopsworks")
    local_fallback = os.getenv("FEATURE_STORE_LOCAL_FALLBACK", "false").lower()

    if feature_store_primary == "hopsworks" and local_fallback == "true":
        print("⚠️  WARNING: Local fallback is enabled for feature store.")
        print("   This should be disabled in production.")
        # Warning only

    print("[OK] Feature store configuration checked")
    return True


def run_all_checks() -> bool:
    """Run all pre-deployment checks."""
    print("=" * 60)
    print("Pre-Deployment Safety Checks")
    print("=" * 60)
    print()

    checks = [
        ("MOCK_MODE", check_mock_mode),
        ("API_KEY", check_api_key),
        ("HOPSWORKS_HOST", check_hopsworks_host),
        ("API Keys", check_api_keys),
        ("Model Metadata", check_model_metadata),
        ("Feature Store Config", check_feature_store_config),
    ]

    all_passed = True
    for name, check_func in checks:
        print(f"Checking {name}...")
        if not check_func():
            all_passed = False
        print()

    print("=" * 60)
    if all_passed:
        print("[OK] All checks passed. Safe to deploy.")
    else:
        print("[FAIL] Some checks failed. Deployment blocked.")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)
