"""
Integration test for backfill pipeline end-to-end.

Tests the complete flow: synthetic data generation → feature engineering
→ target generation → train/val/test split → metadata.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone

from src.data.historical_backfill import (
    generate_mock_historical_dataset,
    verify_api_access,
    collect_sample_data,
)
from src.data.dataset_builder import (
    build_dataset,
    add_source_quality_metadata,
    generate_targets,
)
from src.data.schemas import CityConfig


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def city_configs():
    """Standard city configurations."""
    return [
        CityConfig(id="karachi", name="Karachi", latitude=24.8607, longitude=67.0011),
        CityConfig(id="lahore", name="Lahore", latitude=31.5204, longitude=74.3587),
        CityConfig(id="islamabad", name="Islamabad", latitude=33.6844, longitude=73.0479),
    ]


# =============================================================================
# Integration Tests
# =============================================================================


class TestBackfillPipelineEndToEnd:
    """End-to-end backfill pipeline tests."""

    def test_synthetic_data_generation(self, city_configs):
        """Synthetic data is generated for all cities."""
        df = generate_mock_historical_dataset(
            start_date="2026-08-01",
            end_date="2026-08-04",
            city_configs=city_configs,
        )
        assert len(df) > 0
        assert df["location_id"].nunique() == 3
        # (end - start).total_seconds()/3600 + 1 = 73 hours × 3 cities = 219
        assert len(df) == 219

    def test_synthetic_data_quality(self, city_configs):
        """Synthetic data passes quality checks."""
        df = generate_mock_historical_dataset(
            start_date="2026-08-01",
            end_date="2026-08-04",
            city_configs=city_configs,
        )
        from src.data.validators import full_validation
        report = full_validation(df)
        assert report.status.value in ("pass", "warning")

    def test_full_pipeline_synthetic(self, city_configs):
        """Complete pipeline works with synthetic data."""
        # Generate synthetic data
        df = generate_mock_historical_dataset(
            start_date="2026-08-01",
            end_date="2026-08-07",  # 7 days
            city_configs=city_configs,
        )

        # Build dataset
        result = build_dataset(df, save=False)

        assert "train" in result
        assert "val" in result
        assert "test" in result
        assert result["metadata"]["total_records"] == len(df)
        assert result["metadata"]["feature_count"] > 20
        assert result["metadata"]["target_count"] == 3

    def test_source_quality_metadata(self, city_configs):
        """Source quality metadata is correctly generated."""
        df = generate_mock_historical_dataset(
            start_date="2026-08-01",
            end_date="2026-08-04",
            city_configs=city_configs,
        )
        df = add_source_quality_metadata(df)
        assert "weather_available" in df.columns
        assert "aqi_available" in df.columns
        assert "sources_used" in df.columns

    def test_target_generation_end_to_end(self, city_configs):
        """Target generation works end-to-end."""
        df = generate_mock_historical_dataset(
            start_date="2026-08-01",
            end_date="2026-08-07",
            city_configs=city_configs,
        )
        df = generate_targets(df)
        assert "target_aqi_24h" in df.columns
        assert "target_aqi_48h" in df.columns
        assert "target_aqi_72h" in df.columns

    def test_chronological_split_preserved(self, city_configs):
        """Chronological ordering preserved in splits."""
        df = generate_mock_historical_dataset(
            start_date="2026-08-01",
            end_date="2026-08-07",
            city_configs=city_configs,
        )
        result = build_dataset(df, save=False)
        for split_name in ["train", "val", "test"]:
            split = result[split_name]
            if len(split) > 1:
                assert split["timestamp"].is_monotonic_increasing

    def test_no_cross_city_leakage(self, city_configs):
        """Feature engineering is independent per city."""
        df = generate_mock_historical_dataset(
            start_date="2026-08-01",
            end_date="2026-08-07",
            city_configs=city_configs,
        )
        result = build_dataset(df, save=False)

        # The train split starts at the earliest timestamp, so its first
        # row per city must have NaN lag (no prior data). Val/test splits
        # start later and legitimately have valid lag values.
        train = result["train"]
        for city_id in ["karachi", "lahore", "islamabad"]:
            city_data = train[train["location_id"] == city_id].sort_values("timestamp")
            if len(city_data) > 0 and "aqi_lag_1h" in city_data.columns:
                first_row = city_data.iloc[0]
                assert pd.isna(first_row["aqi_lag_1h"])

        # Verify lag values differ between cities (no cross-city copy)
        for city_id_a, city_id_b in [("karachi", "lahore"), ("lahore", "islamabad")]:
            ca = train[train["location_id"] == city_id_a].sort_values("timestamp")
            cb = train[train["location_id"] == city_id_b].sort_values("timestamp")
            if len(ca) > 1 and len(cb) > 1:
                # Second row lag values should differ between cities
                assert ca.iloc[1]["aqi_lag_1h"] != cb.iloc[1]["aqi_lag_1h"]

    def test_verify_api_access(self):
        """API access verification completes without error."""
        result = verify_api_access()
        assert "openweather" in result
        assert "aqicn" in result
        assert "limitations" in result["openweather"]

    def test_dataset_metadata_complete(self, city_configs):
        """Dataset metadata contains all required fields."""
        df = generate_mock_historical_dataset(
            start_date="2026-08-01",
            end_date="2026-08-04",
            city_configs=city_configs,
        )
        result = build_dataset(df, save=False)
        meta = result["metadata"]
        required_keys = [
            "dataset_version", "feature_version", "schema_version",
            "generation_timestamp", "total_records", "train_records",
            "val_records", "test_records", "feature_count", "target_count",
            "cities", "leakage_errors", "quality_report",
        ]
        for key in required_keys:
            assert key in meta, f"Missing metadata key: {key}"
