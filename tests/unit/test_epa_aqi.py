"""
Deterministic tests for US EPA AQI calculation.

Tests use known EPA breakpoint values and expected AQI results.
Reference: https://aqs.epa.gov/aqsweb/documents/codetables/aqi_breakpoints.html
PM2.5 breakpoints updated May 2024: Good 0.0-9.0 ug/m3 (was 0.0-12.0).
"""
import pytest
import math

from src.utils.epa_aqi import (
    calculate_pm25_aqi,
    calculate_pm10_aqi,
    calculate_o3_aqi,
    calculate_no2_aqi,
    calculate_so2_aqi,
    calculate_co_aqi,
    calculate_individual_aqi,
    calculate_aqi_from_concentration,
    ug_m3_to_ppm,
    ug_m3_to_ppb,
    PM25_BREAKPOINTS,
    PM10_BREAKPOINTS,
    O3_BREAKPOINTS_PPM,
    NO2_BREAKPOINTS_PPB,
    SO2_BREAKPOINTS_PPB,
    CO_BREAKPOINTS_PPM,
    AQI_CALCULATION_VERSION,
    get_aqi_metadata,
)


# ============================================================================
# PM2.5 Tests (Updated May 2024 breakpoints)
# ============================================================================


class TestPM25AQI:
    """Test PM2.5 AQI calculation with 2024 breakpoints."""

    def test_good_lower_bound(self):
        """PM2.5 = 0.0 ug/m3 -> AQI = 0 (Good)."""
        assert calculate_pm25_aqi(0.0) == 0

    def test_good_upper_bound(self):
        """PM2.5 = 9.0 ug/m3 -> AQI = 50 (Good)."""
        assert calculate_pm25_aqi(9.0) == 50

    def test_moderate_lower_bound(self):
        """PM2.5 = 9.1 ug/m3 -> AQI = 51 (Moderate)."""
        assert calculate_pm25_aqi(9.1) == 51

    def test_moderate_upper_bound(self):
        """PM2.5 = 35.4 ug/m3 -> AQI = 100 (Moderate)."""
        assert calculate_pm25_aqi(35.4) == 100

    def test_unhealthy_sensitive_lower(self):
        """PM2.5 = 35.5 ug/m3 -> AQI = 101 (Unhealthy for Sensitive)."""
        assert calculate_pm25_aqi(35.5) == 101

    def test_unhealthy_lower(self):
        """PM2.5 = 55.5 ug/m3 -> AQI = 151 (Unhealthy)."""
        assert calculate_pm25_aqi(55.5) == 151

    def test_very_unhealthy_lower(self):
        """PM2.5 = 125.5 ug/m3 -> AQI = 201 (Very Unhealthy)."""
        assert calculate_pm25_aqi(125.5) == 201

    def test_hazardous_lower(self):
        """PM2.5 = 225.5 ug/m3 -> AQI = 301 (Hazardous)."""
        assert calculate_pm25_aqi(225.5) == 301

    def test_between_breakpoints(self):
        """PM2.5 = 20.0 ug/m3 -> AQI between 50 and 100."""
        aqi = calculate_pm25_aqi(20.0)
        assert 50 < aqi < 100

    def test_known_epa_example(self):
        """PM2.5 = 16.77 ug/m3 (from OpenWeather) -> AQI should be ~59."""
        aqi = calculate_pm25_aqi(16.77)
        # 16.77 is between 9.1 (AQI 51) and 35.4 (AQI 100)
        # AQI = ((100-51)/(35.4-9.1)) * (16.77-9.1) + 51 = 65
        assert 55 <= aqi <= 70

    def test_none_returns_none(self):
        """None concentration returns None AQI."""
        assert calculate_pm25_aqi(None) is None

    def test_nan_returns_none(self):
        """NaN concentration returns None AQI."""
        assert calculate_pm25_aqi(float("nan")) is None


# ============================================================================
# PM10 Tests
# ============================================================================


class TestPM10AQI:
    """Test PM10 AQI calculation."""

    def test_good_upper_bound(self):
        """PM10 = 54.0 ug/m3 -> AQI = 50 (Good)."""
        assert calculate_pm10_aqi(54.0) == 50

    def test_moderate_upper_bound(self):
        """PM10 = 154.0 ug/m3 -> AQI = 100 (Moderate)."""
        assert calculate_pm10_aqi(154.0) == 100

    def test_known_value(self):
        """PM10 = 70.19 ug/m3 -> AQI should be ~56."""
        aqi = calculate_pm10_aqi(70.19)
        assert 50 <= aqi <= 60


# ============================================================================
# O3 Tests (8-hour, converted from ug/m3 to ppm)
# ============================================================================


class TestO3AQI:
    """Test O3 AQI calculation with unit conversion."""

    def test_conversion_correct(self):
        """Verify ug/m3 to ppm conversion for O3."""
        # 100 ug/m3 O3 = (100 * 24.45) / (48 * 1000) = 0.0509 ppm
        ppm = ug_m3_to_ppm(100.0, 48.0)
        assert abs(ppm - 0.0509) < 0.001

    def test_good_upper_bound(self):
        """O3 breakpoint: 0.054 ppm = 106.0 ug/m3 -> AQI = 50 (Good)."""
        # 0.054 ppm = 0.054 * 48 / 24.45 * 1e6 = 106.0 ug/m3
        aqi = calculate_o3_aqi(106.0)
        assert aqi == 50

    def test_moderate_range(self):
        """O3 in moderate range."""
        # 0.055 ppm = 108.0 ug/m3
        aqi = calculate_o3_aqi(108.0)
        assert 50 < aqi <= 100


# ============================================================================
# NO2 Tests (1-hour, converted from ug/m3 to ppb)
# ============================================================================


class TestNO2AQI:
    """Test NO2 AQI calculation with unit conversion."""

    def test_conversion_correct(self):
        """Verify ug/m3 to ppb conversion for NO2."""
        # 100 ug/m3 NO2 = (100 * 24.45) / 46 = 53.15 ppb
        ppb = ug_m3_to_ppb(100.0, 46.0)
        assert abs(ppb - 53.15) < 0.1

    def test_good_upper_bound(self):
        """NO2 = 99.2 ug/m3 -> 53.0 ppb -> AQI = 50 (Good)."""
        # 53.0 ppb = 53.0 * 46 / 24.45 = 99.2 ug/m3
        aqi = calculate_no2_aqi(99.2)
        assert aqi == 50

    def test_moderate_range(self):
        """NO2 in moderate range."""
        # 102.0 ug/m3 = 54.2 ppb -> AQI = 51
        aqi = calculate_no2_aqi(102.0)
        assert aqi is not None
        assert aqi >= 51


# ============================================================================
# SO2 Tests
# ============================================================================


class TestSO2AQI:
    """Test SO2 AQI calculation with unit conversion."""

    def test_conversion_correct(self):
        """Verify ug/m3 to ppb conversion for SO2."""
        # 100 ug/m3 SO2 = (100 * 24.45) / 64 = 38.2 ppb
        ppb = ug_m3_to_ppb(100.0, 64.0)
        assert abs(ppb - 38.2) < 0.1

    def test_good_upper_bound(self):
        """SO2 = 85.4 ug/m3 -> 35.0 ppb -> AQI = 50 (Good)."""
        # 35.0 ppb = 35.0 * 64 / 24.45 = 91.6 ug/m3
        aqi = calculate_so2_aqi(91.6)
        assert aqi == 50


# ============================================================================
# CO Tests
# ============================================================================


class TestCOAQI:
    """Test CO AQI calculation with unit conversion."""

    def test_conversion_correct(self):
        """Verify ug/m3 to ppm conversion for CO."""
        # 5000 ug/m3 CO = 5000 * 24.45 / 28 = 4366 ppm? No.
        # ppm = ug_m3 * 24.45 / M = 5000 * 24.45 / 28 = 4366? That's wrong.
        # 1 ppm = M/24.45 ug/m3 = 28/24.45 = 1.145 ug/m3
        # So 5000 ug/m3 = 5000 / 1.145 = 4366 ppm? No, that's too high.
        # Wait: 1 ppm = (M/24.45) ug/m3 = (28/24.45) = 1.145 ug/m3
        # So 5000 ug/m3 = 5000 / 1.145 = 4366 ppm? That can't be right.
        # Let me reconsider: at 25C, 1 atm, 1 mole = 24.45 L
        # 1 ppm = 1 part per million by volume = 1e-6 mol/mol
        # Concentration in ug/m3 = (ppm * M * 1000) / 24.45
        # So ppm = (conc_ug_m3 * 24.45) / (M * 1000)
        ppm = ug_m3_to_ppm(5000.0, 28.0)
        # (5000 * 24.45) / (28 * 1000) = 4.366 ppm
        assert abs(ppm - 4.366) < 0.01

    def test_good_upper_bound(self):
        """CO = 5038 ug/m3 -> 4.4 ppm -> AQI = 50 (Good)."""
        # 4.4 ppm = 4.4 * 28 / 24.45 * 1000 = 5038 ug/m3
        aqi = calculate_co_aqi(5038.0)
        assert aqi == 50


# ============================================================================
# Dominant Pollutant & Overall AQI
# ============================================================================


class TestDominantPollutant:
    """Test dominant pollutant identification."""

    def test_pm25_dominant(self):
        """PM2.5 is dominant when highest AQI."""
        aqi, dominant = calculate_individual_aqi(pm25=200.0, pm10=50.0, o3=50.0)
        assert aqi is not None
        assert dominant == "pm25"

    def test_pm10_dominant(self):
        """PM10 is dominant when highest AQI."""
        aqi, dominant = calculate_individual_aqi(pm25=5.0, pm10=200.0)
        assert aqi is not None
        assert dominant == "pm10"

    def test_all_none(self):
        """All None returns None."""
        aqi, dominant = calculate_individual_aqi()
        assert aqi is None
        assert dominant is None

    def test_mixed_data(self):
        """Realistic mixed pollutant data."""
        aqi, dominant = calculate_individual_aqi(
            pm25=16.77, pm10=70.19, o3=44.14, no2=0.08, so2=0.39, co=70.96
        )
        assert aqi is not None
        # PM2.5 dominates at these levels
        assert dominant == "pm25"


# ============================================================================
# Category Transitions
# ============================================================================


class TestCategoryTransitions:
    """Test AQI category boundaries."""

    def test_good_to_moderate_pm25(self):
        """PM2.5 boundary: 9.0 (AQI 50) -> 9.1 (AQI 51)."""
        assert calculate_pm25_aqi(9.0) == 50
        assert calculate_pm25_aqi(9.1) == 51

    def test_moderate_to_unhealthy_sensitive_pm25(self):
        """PM2.5 boundary: 35.4 (AQI 100) -> 35.5 (AQI 101)."""
        assert calculate_pm25_aqi(35.4) == 100
        assert calculate_pm25_aqi(35.5) == 101

    def test_pm25_aqi_500_cap(self):
        """PM2.5 above highest breakpoint caps at 500."""
        assert calculate_pm25_aqi(500.0) == 500
        assert calculate_pm25_aqi(1000.0) == 500


# ============================================================================
# Metadata
# ============================================================================


class TestAQIMetadata:
    """Test AQI calculation metadata."""

    def test_metadata_structure(self):
        """Metadata contains required fields."""
        meta = get_aqi_metadata()
        assert meta["aqi_standard"] == "US_EPA"
        assert meta["aqi_derived"] is True
        assert meta["aqi_source"] == "openweather_pollutants"
        assert "2024" in meta["aqi_method_version"]

    def test_version_in_metadata(self):
        """Version is documented for audit trail."""
        meta = get_aqi_metadata()
        assert "pm25_breakpoint_note" in meta
        assert "9.0" in meta["pm25_breakpoint_note"]
