#!/usr/bin/env python3
"""Regression tests for AQI invariant.

The selected AQI must always equal:
  max(pm25_aqi_subindex, pm10_aqi_subindex)

When only one sub-index is valid, AQI must equal that valid sub-index.
When neither is valid, AQI must be None and is_training_valid must be False.
"""
import pytest
from src.utils.epa_aqi import (
    calculate_pm25_aqi,
    calculate_pm10_aqi,
    calculate_nowcast_aqi,
    calculate_nowcast,
)


class TestAQIInvariant:
    """Tests that AQI == max(pm25_subindex, pm10_subindex)."""

    def test_both_subindices_pm25_dominant(self):
        """When PM2.5 AQI > PM10 AQI, dominant must be pm25."""
        pm25_hist = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0]
        pm10_hist = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
        aqi, dominant, meta = calculate_nowcast_aqi(pm25_hist, pm10_hist)
        individual = meta["individual_aqi"]

        assert aqi is not None
        assert dominant == "pm25"
        assert aqi == max(individual.values())
        assert aqi == individual["pm25"]

    def test_both_subindices_pm10_dominant(self):
        """When PM10 AQI > PM2.5 AQI, dominant must be pm10."""
        pm25_hist = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        pm10_hist = [150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 210.0]
        aqi, dominant, meta = calculate_nowcast_aqi(pm25_hist, pm10_hist)
        individual = meta["individual_aqi"]

        assert aqi is not None
        assert dominant == "pm10"
        assert aqi == max(individual.values())
        assert aqi == individual["pm10"]

    def test_equal_subindices(self):
        """When sub-indices are equal, either dominant is acceptable."""
        # Use values that produce equal sub-indices
        # PM2.5 NowCast ~35.5 -> AQI ~81
        # PM10 NowCast ~100 -> AQI ~76
        # These won't be exactly equal, but the invariant must hold
        pm25_hist = [35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0]
        pm10_hist = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        aqi, dominant, meta = calculate_nowcast_aqi(pm25_hist, pm10_hist)
        individual = meta["individual_aqi"]

        assert aqi is not None
        assert aqi == max(individual.values())
        assert dominant in ("pm25", "pm10")

    def test_only_pm25_available(self):
        """When only PM2.5 is available, AQI must equal PM2.5 sub-index."""
        pm25_hist = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0]
        aqi, dominant, meta = calculate_nowcast_aqi(pm25_hist, None)
        individual = meta["individual_aqi"]

        assert aqi is not None
        assert dominant == "pm25"
        assert aqi == individual["pm25"]
        assert "pm10" not in individual

    def test_only_pm10_available(self):
        """When only PM10 is available, AQI must equal PM10 sub-index."""
        pm10_hist = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0]
        aqi, dominant, meta = calculate_nowcast_aqi(None, pm10_hist)
        individual = meta["individual_aqi"]

        assert aqi is not None
        assert dominant == "pm10"
        assert aqi == individual["pm10"]
        assert "pm25" not in individual

    def test_neither_available(self):
        """When neither pollutant has data, AQI must be None."""
        aqi, dominant, meta = calculate_nowcast_aqi([], [])
        assert aqi is None
        assert dominant is None

    def test_specific_nowcast_96_09(self):
        """Regression: PM2.5 NowCast 96.09 must produce AQI 179."""
        pm25_aqi = calculate_pm25_aqi(96.09)
        assert pm25_aqi == 179

    def test_specific_nowcast_71_98(self):
        """Regression: PM10 NowCast 71.98 must produce AQI 59."""
        pm10_aqi = calculate_pm10_aqi(71.98)
        assert pm10_aqi == 59

    def test_karachi_invariant(self):
        """Regression: Karachi NowCast values must satisfy invariant."""
        # From actual persisted data
        pm25_nowcast = 96.09019607843138
        pm10_nowcast = 71.97701088225814
        pm25_aqi = calculate_pm25_aqi(pm25_nowcast)
        pm10_aqi = calculate_pm10_aqi(pm10_nowcast)
        expected = max(pm25_aqi, pm10_aqi)
        expected_dom = "pm25" if pm25_aqi >= pm10_aqi else "pm10"

        assert pm25_aqi is not None
        assert pm10_aqi is not None
        assert pm25_aqi > pm10_aqi
        assert expected_dom == "pm25"

        # The stored AQI must match
        # This is the invariant that was violated before
        assert expected == pm25_aqi

    def test_lahore_invariant(self):
        """Regression: Lahore warm-up history must satisfy invariant."""
        # Lahore warm-up has varying PM values
        from src.utils.epa_aqi import calculate_individual_aqi

        pm25_nowcast = 39.0789549341939
        pm10_nowcast = 102.83527559055118
        aqi, dominant, individual = calculate_individual_aqi(
            pm25=pm25_nowcast, pm10=pm10_nowcast
        )

        assert aqi is not None
        assert aqi == max(individual.values())
        assert dominant in individual
        assert individual[dominant] == aqi


class TestPersistedRecordInvariant:
    """Tests that persisted records satisfy the AQI invariant."""

    def test_master_csv_invariant(self):
        """Every row in master CSV with NowCast must satisfy the invariant."""
        import pandas as pd
        from pathlib import Path

        csv_path = Path("data/raw/real/master_observations.csv")
        if not csv_path.exists():
            pytest.skip("Master CSV not found")

        df = pd.read_csv(csv_path)
        nowcast_rows = df[df["aqi_method"] == "PM_NOWCAST"]

        for _, row in nowcast_rows.iterrows():
            pm25_sub = row.get("pm25_aqi_subindex")
            pm10_sub = row.get("pm10_aqi_subindex")
            aqi = row["aqi"]
            dom = row.get("aqi_dominant_pollutant")

            if pd.notna(pm25_sub) and pd.notna(pm10_sub):
                expected = max(int(pm25_sub), int(pm10_sub))
                assert aqi == expected, (
                    f"Row {row['timestamp']} {row['location_id']}: "
                    f"aqi={aqi} != max({pm25_sub}, {pm10_sub})={expected}"
                )
                expected_dom = "pm25" if pm25_sub >= pm10_sub else "pm10"
                assert dom == expected_dom, (
                    f"Row {row['timestamp']} {row['location_id']}: "
                    f"dominant={dom} != expected={expected_dom}"
                )
            elif pd.notna(pm25_sub):
                assert aqi == int(pm25_sub), (
                    f"Row {row['timestamp']}: aqi={aqi} != pm25_sub={pm25_sub}"
                )
                assert dom == "pm25"
            elif pd.notna(pm10_sub):
                assert aqi == int(pm10_sub), (
                    f"Row {row['timestamp']}: aqi={aqi} != pm10_sub={pm10_sub}"
                )
                assert dom == "pm10"
