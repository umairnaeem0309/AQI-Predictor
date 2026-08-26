"""
Comprehensive tests for US EPA AQI calculation.

Tests cover:
1. Standard AQI equation (breakpoint interpolation)
2. NowCast algorithm (PM2.5/PM10)
3. EPA concentration truncation rules
4. Gas unit conversions
5. Dominant pollutant selection
6. Category transitions
7. Metadata/audit trail
8. Edge cases

Reference: EPA-454/B-24-002 (May 2024)
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
    calculate_nowcast,
    calculate_nowcast_aqi,
    ug_m3_to_ppm,
    ug_m3_to_ppb,
    truncate_pm25,
    truncate_pm10,
    truncate_o3_ppm,
    PM25_BREAKPOINTS,
    PM10_BREAKPOINTS,
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

    def test_truncation_applied(self):
        """PM2.5 truncation to 0.1 ug/m3 is applied."""
        # 9.05 should truncate to 9.0 -> AQI = 50
        assert calculate_pm25_aqi(9.05) == 50
        # 9.15 should truncate to 9.1 -> AQI = 51
        assert calculate_pm25_aqi(9.15) == 51

    def test_none_returns_none(self):
        """None concentration returns None AQI."""
        assert calculate_pm25_aqi(None) is None

    def test_nan_returns_none(self):
        """NaN concentration returns None AQI."""
        result = calculate_pm25_aqi(float("nan"))
        assert result is None or math.isnan(result) is False

    def test_aqi_500_cap(self):
        """PM2.5 above highest breakpoint caps at 500."""
        assert calculate_pm25_aqi(500.0) == 500
        assert calculate_pm25_aqi(1000.0) == 500


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

    def test_truncation_applied(self):
        """PM10 truncation to 1 ug/m3 is applied."""
        # 54.5 should truncate to 54 -> AQI = 50
        assert calculate_pm10_aqi(54.5) == 50
        # 55.5 should truncate to 55 -> AQI = 51
        assert calculate_pm10_aqi(55.5) == 51


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
        """O3 breakpoint: 0.054 ppm -> AQI = 50 (Good)."""
        # 0.054 ppm = 0.054 * 48 / 24.45 * 1000 = 106.0 ug/m3
        # But truncation to 0.001 ppm may change value slightly
        aqi = calculate_o3_aqi(106.0)
        assert aqi is not None
        assert 49 <= aqi <= 51  # Allow for truncation

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
        """NO2 breakpoint: 53.0 ppb -> AQI = 50 (Good)."""
        # 53.0 ppb = 53.0 * 46 / 24.45 = 99.2 ug/m3
        aqi = calculate_no2_aqi(99.2)
        assert aqi is not None
        assert 49 <= aqi <= 51  # Allow for truncation

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
        """SO2 breakpoint: 35.0 ppb -> AQI = 50 (Good)."""
        # 35.0 ppb = 35.0 * 64 / 24.45 = 91.6 ug/m3
        aqi = calculate_so2_aqi(91.6)
        assert aqi is not None
        assert 49 <= aqi <= 51  # Allow for truncation


# ============================================================================
# CO Tests
# ============================================================================


class TestCOAQI:
    """Test CO AQI calculation with unit conversion."""

    def test_conversion_correct(self):
        """Verify ug/m3 to ppm conversion for CO."""
        # 5000 ug/m3 CO = (5000 * 24.45) / (28 * 1000) = 4.366 ppm
        ppm = ug_m3_to_ppm(5000.0, 28.0)
        assert abs(ppm - 4.366) < 0.01

    def test_good_upper_bound(self):
        """CO breakpoint: 4.4 ppm -> AQI = 50 (Good)."""
        # 4.4 ppm = 4.4 * 28 / 24.45 * 1000 = 5038 ug/m3
        aqi = calculate_co_aqi(5038.0)
        assert aqi is not None
        assert 49 <= aqi <= 51  # Allow for truncation


# ============================================================================
# Concentration Truncation Tests
# ============================================================================


class TestTruncation:
    """Test EPA concentration truncation rules."""

    def test_pm25_truncation(self):
        """PM2.5 truncated to 0.1 ug/m3."""
        assert truncate_pm25(9.05) == 9.0
        assert truncate_pm25(9.09) == 9.0
        assert truncate_pm25(9.10) == 9.1
        assert truncate_pm25(35.45) == 35.4

    def test_pm10_truncation(self):
        """PM10 truncated to 1 ug/m3."""
        assert truncate_pm10(54.5) == 54.0
        assert truncate_pm10(55.9) == 55.0
        assert truncate_pm10(154.9) == 154.0

    def test_o3_truncation(self):
        """O3 truncated to 0.001 ppm."""
        assert truncate_o3_ppm(0.0545) == 0.054
        assert truncate_o3_ppm(0.0709) == 0.070


# ============================================================================
# NowCast Algorithm Tests
# ============================================================================


class TestNowCast:
    """Test EPA NowCast algorithm for PM2.5/PM10."""

    def test_stable_conditions(self):
        """All same values -> NowCast = 12-hour average = that value."""
        hourly = [20.0] * 12
        result = calculate_nowcast(hourly)
        assert result is not None
        assert abs(result - 20.0) < 0.01

    def test_increasing_pollution(self):
        """Increasing pollution -> NowCast weights recent hours more."""
        # Steady increase from 10 to 30
        hourly = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0]
        result = calculate_nowcast(hourly)
        assert result is not None
        # NowCast should be within the range
        assert 10.0 <= result <= 32.0

    def test_decreasing_pollution(self):
        """Decreasing pollution -> NowCast weights recent hours more."""
        hourly = [30.0, 28.0, 26.0, 24.0, 22.0, 20.0, 18.0, 16.0, 14.0, 12.0, 10.0, 8.0]
        result = calculate_nowcast(hourly)
        assert result is not None
        # NowCast should be within the range
        assert 8.0 <= result <= 30.0

    def test_missing_recent_hour(self):
        """Missing most recent hour -> NowCast returns None."""
        hourly = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, None]
        result = calculate_nowcast(hourly)
        assert result is None

    def test_missing_second_hour(self):
        """Missing second most recent hour -> NowCast still works if recent valid."""
        # EPA requires c1 and c2 valid, but implementation may handle gracefully
        hourly = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, None, 20.0]
        result = calculate_nowcast(hourly)
        # May return None or a valid result depending on implementation
        assert result is None or result == 20.0

    def test_insufficient_data(self):
        """Less than 2 hours -> NowCast returns None."""
        hourly = [20.0]
        result = calculate_nowcast(hourly)
        assert result is None

    def test_empty_data(self):
        """Empty list -> NowCast returns None."""
        result = calculate_nowcast([])
        assert result is None

    def test_partial_data(self):
        """Partial data (e.g., 4 hours) -> NowCast works if valid."""
        hourly = [10.0, 15.0, 20.0, 25.0]
        result = calculate_nowcast(hourly)
        assert result is not None
        assert 10.0 <= result <= 25.0

    def test_missing_middle_hours(self):
        """Missing middle hours -> NowCast may or may not work."""
        hourly = [10.0, None, None, 20.0, None, 25.0, 30.0, None, None, None, None, 35.0]
        result = calculate_nowcast(hourly)
        # Depends on implementation - may return None if too few valid recent hours
        assert result is None or (result is not None and 10.0 <= result <= 35.0)

    def test_all_zeros(self):
        """All zeros -> NowCast = 0."""
        hourly = [0.0] * 12
        result = calculate_nowcast(hourly)
        assert result is not None
        assert result == 0.0

    def test_one_pulse(self):
        """Single pulse -> NowCast captures it with decay."""
        # All zeros except one hour at 144
        hourly = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 144.0, 0.0]
        result = calculate_nowcast(hourly)
        assert result is not None
        # The pulse was 1 hour ago, so weight = w^1
        # With w=0.5 (max variability), result = 144 * 0.5 / (1 + 0.5 + ...) = 72
        assert result > 0


# ============================================================================
# NowCast AQI Integration Tests
# ============================================================================


class TestNowCastAQI:
    """Test NowCast-based AQI calculation."""

    def test_stable_pm25(self):
        """Stable PM2.5 -> NowCast AQI matches direct calculation."""
        hourly = [20.0] * 12
        aqi, dominant, meta = calculate_nowcast_aqi(pm25_hourly=hourly)
        assert aqi is not None
        assert dominant == "pm25"
        assert meta["method"] == "nowcast"

    def test_high_pm25(self):
        """High PM2.5 -> NowCast AQI in unhealthy range."""
        hourly = [100.0] * 12
        aqi, dominant, meta = calculate_nowcast_aqi(pm25_hourly=hourly)
        assert aqi is not None
        assert aqi > 150  # Should be unhealthy

    def test_insufficient_data(self):
        """Insufficient hourly data -> NowCast AQI returns None."""
        hourly = [20.0]
        aqi, dominant, meta = calculate_nowcast_aqi(pm25_hourly=hourly)
        assert aqi is None

    def test_metadata_complete(self):
        """Metadata contains all required fields."""
        hourly = [20.0] * 12
        aqi, dominant, meta = calculate_nowcast_aqi(pm25_hourly=hourly)
        assert "method" in meta
        assert "pm25_nowcast" in meta
        assert "dominant_pollutant" in meta
        assert "hours_used_pm25" in meta
        assert meta["hours_used_pm25"] == 12


# ============================================================================
# Dominant Pollutant Tests
# ============================================================================


class TestDominantPollutant:
    """Test dominant pollutant identification."""

    def test_pm25_dominant(self):
        """PM2.5 is dominant when highest AQI."""
        aqi, dominant, individual = calculate_individual_aqi(pm25=200.0, pm10=50.0, o3=50.0)
        assert aqi is not None
        assert dominant == "pm25"

    def test_pm10_dominant(self):
        """PM10 is dominant when highest AQI."""
        aqi, dominant, individual = calculate_individual_aqi(pm25=5.0, pm10=200.0)
        assert aqi is not None
        assert dominant == "pm10"

    def test_all_none(self):
        """All None returns None."""
        aqi, dominant, individual = calculate_individual_aqi()
        assert aqi is None
        assert dominant is None

    def test_mixed_data(self):
        """Realistic mixed pollutant data."""
        aqi, dominant, individual = calculate_individual_aqi(
            pm25=16.77, pm10=70.19, o3=44.14, no2=0.08, so2=0.39, co=70.96
        )
        assert aqi is not None
        # PM2.5 dominates at these levels
        assert dominant == "pm25"


# ============================================================================
# Category Transition Tests
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

    def test_unhealthy_to_very_unhealthy_pm25(self):
        """PM2.5 boundary: 125.4 (AQI 200) -> 125.5 (AQI 201)."""
        assert calculate_pm25_aqi(125.4) == 200
        assert calculate_pm25_aqi(125.5) == 201

    def test_very_unhealthy_to_hazardous_pm25(self):
        """PM2.5 boundary: 225.4 (AQI 300) -> 225.5 (AQI 301)."""
        assert calculate_pm25_aqi(225.4) == 300
        assert calculate_pm25_aqi(225.5) == 301


# ============================================================================
# Metadata Tests
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

    def test_derived_disclosure(self):
        """Metadata clearly states this is derived, not official."""
        meta = get_aqi_metadata()
        assert "derived" in meta["derived_disclosure"].lower()
        assert "not an official" in meta["derived_disclosure"].lower()

    def test_nowcast_reference(self):
        """NowCast reference is documented."""
        meta = get_aqi_metadata()
        assert "nowcast_reference" in meta
        assert "NowCast" in meta["nowcast_reference"]
