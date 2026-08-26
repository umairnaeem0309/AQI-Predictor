"""
US EPA AQI Calculation from Pollutant Concentrations.

Implements:
1. Standard AQI equation (linear interpolation between breakpoints)
2. NowCast algorithm for PM2.5 and PM10 (real-time hourly AQI)
3. Unit conversions for gas pollutants (ug/m3 -> ppm/ppb)
4. Dominant pollutant selection
5. Full metadata for audit trail

References:
- EPA AQI Breakpoints (May 2024): https://aqs.epa.gov/aqsweb/documents/codetables/aqi_breakpoints.html
- EPA NowCast Methodology: https://en.wikipedia.org/wiki/NowCast_(air_quality_index)
- EPA Technical Assistance Document: EPA-454/B-24-002 (May 2024)

IMPORTANT: This calculates a DERIVED EPA-method AQI estimate from OpenWeather
pollutant concentrations. It is NOT an official EPA/AirNow monitor reading.
"""
import math
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# ============================================================================
# EPA AQI Breakpoint Tables (Current as of May 2024)
# Source: https://aqs.epa.gov/aqsweb/documents/codetables/aqi_breakpoints.html
# ============================================================================

@dataclass
class BreakpointRow:
    """A single AQI breakpoint row."""
    pollutant: str
    aqi_low: int
    aqi_high: int
    conc_low: float
    conc_high: float
    unit: str
    averaging_period: str


# PM2.5 (ug/m3, 24-hour average) — Updated May 2024
PM25_BREAKPOINTS: List[BreakpointRow] = [
    BreakpointRow("pm25", 0, 50, 0.0, 9.0, "ug/m3", "24-hour"),
    BreakpointRow("pm25", 51, 100, 9.1, 35.4, "ug/m3", "24-hour"),
    BreakpointRow("pm25", 101, 150, 35.5, 55.4, "ug/m3", "24-hour"),
    BreakpointRow("pm25", 151, 200, 55.5, 125.4, "ug/m3", "24-hour"),
    BreakpointRow("pm25", 201, 300, 125.5, 225.4, "ug/m3", "24-hour"),
    BreakpointRow("pm25", 301, 400, 225.5, 325.4, "ug/m3", "24-hour"),
    BreakpointRow("pm25", 401, 500, 325.5, 500.4, "ug/m3", "24-hour"),
]

# PM10 (ug/m3, 24-hour average)
PM10_BREAKPOINTS: List[BreakpointRow] = [
    BreakpointRow("pm10", 0, 50, 0.0, 54.0, "ug/m3", "24-hour"),
    BreakpointRow("pm10", 51, 100, 55.0, 154.0, "ug/m3", "24-hour"),
    BreakpointRow("pm10", 101, 150, 155.0, 254.0, "ug/m3", "24-hour"),
    BreakpointRow("pm10", 151, 200, 255.0, 354.0, "ug/m3", "24-hour"),
    BreakpointRow("pm10", 201, 300, 355.0, 424.0, "ug/m3", "24-hour"),
    BreakpointRow("pm10", 301, 400, 425.0, 604.0, "ug/m3", "24-hour"),
    BreakpointRow("pm10", 401, 500, 605.0, 749.0, "ug/m3", "24-hour"),
]

# O3 (ppm, 8-hour average)
O3_BREAKPOINTS_PPM: List[BreakpointRow] = [
    BreakpointRow("o3", 0, 50, 0.0, 0.054, "ppm", "8-hour"),
    BreakpointRow("o3", 51, 100, 0.055, 0.070, "ppm", "8-hour"),
    BreakpointRow("o3", 101, 150, 0.071, 0.085, "ppm", "8-hour"),
    BreakpointRow("o3", 151, 200, 0.086, 0.105, "ppm", "8-hour"),
    BreakpointRow("o3", 201, 300, 0.106, 0.200, "ppm", "8-hour"),
]

# NO2 (ppb, 1-hour)
NO2_BREAKPOINTS_PPB: List[BreakpointRow] = [
    BreakpointRow("no2", 0, 50, 0.0, 53.0, "ppb", "1-hour"),
    BreakpointRow("no2", 51, 100, 54.0, 100.0, "ppb", "1-hour"),
    BreakpointRow("no2", 101, 150, 101.0, 360.0, "ppb", "1-hour"),
    BreakpointRow("no2", 151, 200, 361.0, 649.0, "ppb", "1-hour"),
    BreakpointRow("no2", 201, 300, 650.0, 1249.0, "ppb", "1-hour"),
    BreakpointRow("no2", 301, 400, 1250.0, 2049.0, "ppb", "1-hour"),
    BreakpointRow("no2", 401, 500, 2050.0, 3049.0, "ppb", "1-hour"),
]

# SO2 (ppb, 1-hour)
SO2_BREAKPOINTS_PPB: List[BreakpointRow] = [
    BreakpointRow("so2", 0, 50, 0.0, 35.0, "ppb", "1-hour"),
    BreakpointRow("so2", 51, 100, 36.0, 75.0, "ppb", "1-hour"),
    BreakpointRow("so2", 101, 150, 76.0, 185.0, "ppb", "1-hour"),
    BreakpointRow("so2", 151, 200, 186.0, 304.0, "ppb", "1-hour"),
    BreakpointRow("so2", 201, 300, 305.0, 604.0, "ppb", "1-hour"),
    BreakpointRow("so2", 301, 400, 605.0, 1004.0, "ppb", "1-hour"),
    BreakpointRow("so2", 401, 500, 1005.0, 2004.0, "ppb", "1-hour"),
]

# CO (ppm, 8-hour)
CO_BREAKPOINTS_PPM: List[BreakpointRow] = [
    BreakpointRow("co", 0, 50, 0.0, 4.4, "ppm", "8-hour"),
    BreakpointRow("co", 51, 100, 4.5, 9.4, "ppm", "8-hour"),
    BreakpointRow("co", 101, 150, 9.5, 12.4, "ppm", "8-hour"),
    BreakpointRow("co", 151, 200, 12.5, 15.4, "ppm", "8-hour"),
    BreakpointRow("co", 201, 300, 15.5, 30.4, "ppm", "8-hour"),
    BreakpointRow("co", 301, 400, 30.5, 50.4, "ppm", "8-hour"),
    BreakpointRow("co", 401, 500, 50.5, 60.4, "ppm", "8-hour"),
]

# Metadata
AQI_CALCULATION_VERSION = "US_EPA_2024_May"
AQI_CALCULATION_REFERENCE = "https://aqs.epa.gov/aqsweb/documents/codetables/aqi_breakpoints.html"
AQI_STANDARD = "US_EPA"
NOWCAST_REFERENCE = "https://en.wikipedia.org/wiki/NowCast_(air_quality_index)"


# ============================================================================
# Unit Conversion Functions (Standard Conditions: 25°C, 1 atm)
# ============================================================================

def ug_m3_to_ppm(conc_ug_m3: float, molecular_weight: float) -> float:
    """Convert ug/m3 to ppm at standard conditions (25°C, 1 atm).

    At standard conditions (25°C, 1 atm):
        Molar volume = 24.45 L/mol = 0.02445 m3/mol
        Concentration (ug/m3) = (ppm * M * 1000) / 24.45
        Therefore: ppm = (conc_ug_m3 * 24.45) / (M * 1000)

    Note: This assumes standard temperature and pressure. Historical
    temperature/pressure data is not available from OpenWeather, so
    standard-condition conversion is used. The error is typically <5%
    for ambient conditions.

    Args:
        conc_ug_m3: Concentration in ug/m3
        molecular_weight: Molecular weight of the gas (g/mol)

    Returns:
        Concentration in ppm
    """
    return (conc_ug_m3 * 24.45) / (molecular_weight * 1000)


def ug_m3_to_ppb(conc_ug_m3: float, molecular_weight: float) -> float:
    """Convert ug/m3 to ppb at standard conditions (25°C, 1 atm).

    At standard conditions (25°C, 1 atm):
        Molar volume = 24.45 L/mol = 0.02445 m3/mol
        Concentration (ug/m3) = (ppb * M) / 24.45
        Therefore: ppb = (conc_ug_m3 * 24.45) / M

    Note: This assumes standard temperature and pressure. Historical
    temperature/pressure data is not available from OpenWeather, so
    standard-condition conversion is used. The error is typically <5%
    for ambient conditions.

    Args:
        conc_ug_m3: Concentration in ug/m3
        molecular_weight: Molecular weight of the gas (g/mol)

    Returns:
        Concentration in ppb
    """
    return (conc_ug_m3 * 24.45) / molecular_weight


# ============================================================================
# EPA Concentration Truncation Rules
# ============================================================================

def truncate_pm25(conc: float) -> float:
    """Truncate PM2.5 to 0.1 ug/m3 (EPA rule)."""
    if conc is None or (isinstance(conc, float) and math.isnan(conc)):
        return conc
    return math.floor(conc * 10) / 10


def truncate_pm10(conc: float) -> float:
    """Truncate PM10 to 1 ug/m3 (EPA rule)."""
    return math.floor(conc)


def truncate_o3_ppm(conc_ppm: float) -> float:
    """Truncate O3 to 0.001 ppm = 1 ppb (EPA rule)."""
    return math.floor(conc_ppm * 1000) / 1000


def truncate_no2_ppb(conc_ppb: float) -> float:
    """Truncate NO2 to 1 ppb (EPA rule)."""
    return math.floor(conc_ppb)


def truncate_so2_ppb(conc_ppb: float) -> float:
    """Truncate SO2 to 1 ppb (EPA rule)."""
    return math.floor(conc_ppb)


def truncate_co_ppm(conc_ppm: float) -> float:
    """Truncate CO to 0.1 ppm (EPA rule)."""
    return math.floor(conc_ppm * 10) / 10


# ============================================================================
# AQI Calculation (Standard EPA Equation)
# ============================================================================

def calculate_aqi_from_concentration(
    concentration: float,
    breakpoints: List[BreakpointRow],
) -> Optional[int]:
    """Calculate AQI from concentration using breakpoint table.

    Uses the standard EPA linear interpolation formula:
        AQI = ((AQI_high - AQI_low) / (C_high - C_low)) * (C - C_low) + AQI_low

    The resulting AQI is rounded to the nearest whole number (EPA rule).

    Args:
        concentration: Pollutant concentration (already truncated to EPA precision)
        breakpoints: List of breakpoint rows for this pollutant

    Returns:
        AQI value (integer, rounded) or None if out of range
    """
    if concentration is None or math.isnan(concentration):
        return None

    for bp in breakpoints:
        if bp.conc_low <= concentration <= bp.conc_high:
            aqi_range = bp.aqi_high - bp.aqi_low
            conc_range = bp.conc_high - bp.conc_low
            if conc_range == 0:
                return bp.aqi_low
            aqi = ((aqi_range / conc_range) * (concentration - bp.conc_low)) + bp.aqi_low
            return round(aqi)

    # Beyond breakpoint range
    if concentration > breakpoints[-1].conc_high:
        return 500  # Cap at hazardous
    return None


def calculate_pm25_aqi(pm25_ug_m3: float) -> Optional[int]:
    """Calculate AQI from PM2.5 concentration (ug/m3).

    Uses 2024 EPA breakpoints (Good: 0.0-9.0 ug/m3).
    Truncates to 0.1 ug/m3 before calculation.

    Args:
        pm25_ug_m3: PM2.5 concentration in ug/m3

    Returns:
        AQI value (0-500) or None
    """
    if pm25_ug_m3 is None:
        return None
    truncated = truncate_pm25(pm25_ug_m3)
    return calculate_aqi_from_concentration(truncated, PM25_BREAKPOINTS)


def calculate_pm10_aqi(pm10_ug_m3: float) -> Optional[int]:
    """Calculate AQI from PM10 concentration (ug/m3).

    Truncates to 1 ug/m3 before calculation.

    Args:
        pm10_ug_m3: PM10 concentration in ug/m3

    Returns:
        AQI value (0-500) or None
    """
    if pm10_ug_m3 is None:
        return None
    truncated = truncate_pm10(pm10_ug_m3)
    return calculate_aqi_from_concentration(truncated, PM10_BREAKPOINTS)


def calculate_o3_aqi(o3_ug_m3: float) -> Optional[int]:
    """Calculate AQI from O3 concentration (ug/m3).

    Converts ug/m3 to ppm, truncates to 0.001 ppm, then applies
    EPA 8-hour O3 breakpoints.

    Args:
        o3_ug_m3: O3 concentration in ug/m3

    Returns:
        AQI value (0-300 for 8-hour) or None
    """
    if o3_ug_m3 is None:
        return None
    o3_ppm = ug_m3_to_ppm(o3_ug_m3, 48.0)
    truncated = truncate_o3_ppm(o3_ppm)
    return calculate_aqi_from_concentration(truncated, O3_BREAKPOINTS_PPM)


def calculate_no2_aqi(no2_ug_m3: float) -> Optional[int]:
    """Calculate AQI from NO2 concentration (ug/m3).

    Converts ug/m3 to ppb, truncates to 1 ppb, then applies
    EPA 1-hour NO2 breakpoints.

    Args:
        no2_ug_m3: NO2 concentration in ug/m3

    Returns:
        AQI value (0-500) or None
    """
    if no2_ug_m3 is None:
        return None
    no2_ppb = ug_m3_to_ppb(no2_ug_m3, 46.0)
    truncated = truncate_no2_ppb(no2_ppb)
    return calculate_aqi_from_concentration(truncated, NO2_BREAKPOINTS_PPB)


def calculate_so2_aqi(so2_ug_m3: float) -> Optional[int]:
    """Calculate AQI from SO2 concentration (ug/m3).

    Converts ug/m3 to ppb, truncates to 1 ppb, then applies
    EPA 1-hour SO2 breakpoints.

    Args:
        so2_ug_m3: SO2 concentration in ug/m3

    Returns:
        AQI value (0-500) or None
    """
    if so2_ug_m3 is None:
        return None
    so2_ppb = ug_m3_to_ppb(so2_ug_m3, 64.0)
    truncated = truncate_so2_ppb(so2_ppb)
    return calculate_aqi_from_concentration(truncated, SO2_BREAKPOINTS_PPB)


def calculate_co_aqi(co_ug_m3: float) -> Optional[int]:
    """Calculate AQI from CO concentration (ug/m3).

    Converts ug/m3 to ppm, truncates to 0.1 ppm, then applies
    EPA 8-hour CO breakpoints.

    Args:
        co_ug_m3: CO concentration in ug/m3

    Returns:
        AQI value (0-500) or None
    """
    if co_ug_m3 is None:
        return None
    co_ppm = ug_m3_to_ppm(co_ug_m3, 28.0)
    truncated = truncate_co_ppm(co_ppm)
    return calculate_aqi_from_concentration(truncated, CO_BREAKPOINTS_PPM)


# ============================================================================
# NowCast Algorithm (EPA Methodology for PM2.5 and PM10)
# ============================================================================

def calculate_nowcast(
    hourly_concentrations: List[Optional[float]],
    min_hours: int = 2,
) -> Optional[float]:
    """Calculate EPA NowCast concentration from hourly data.

    The NowCast is a weighted average of up to 12 hours of hourly data,
    where more recent hours are weighted more heavily when pollution
    levels are changing.

    Algorithm (from EPA):
    1. Use the most recent 12 hours of data (or available hours)
    2. Calculate weight factor w based on variability:
       w* = c_min / c_max (ratio of min to max in the window)
       w = max(w*, 0.5)  (minimum weight is 0.5)
    3. NowCast = sum(w^(i-1) * c_i) / sum(w^(i-1))
       where c_1 is most recent, c_12 is oldest

    Requirements:
    - At least 2 of the past 3 hourly values must be valid
    - c_1 (most recent) must be valid
    - If c_1 or c_2 is missing, NowCast cannot be calculated

    Args:
        hourly_concentrations: List of hourly concentrations (most recent last).
            Can contain None for missing hours. Up to 12 values.
        min_hours: Minimum valid hours required (default: 2)

    Returns:
        NowCast concentration or None if insufficient data
    """
    if not hourly_concentrations:
        return None

    # Take the most recent 12 hours
    window = hourly_concentrations[-12:]

    # Check if most recent hours are valid (EPA requirement)
    if len(window) < 2 or window[-1] is None:
        return None

    # Filter to valid (non-None) concentrations
    valid = [(i, c) for i, c in enumerate(window) if c is not None]

    if len(valid) < min_hours:
        return None

    # Check that at least 2 of the past 3 hours are valid
    # (indices -1, -2, -3 in the window)
    recent_3 = [c for c in window[-3:] if c is not None]
    if len(recent_3) < 2:
        return None

    # Extract valid concentrations and their positions
    positions = [i for i, c in valid]
    concentrations = [c for i, c in valid]

    # Calculate weight factor
    c_min = min(concentrations)
    c_max = max(concentrations)

    if c_max == 0:
        # All zeros — no variability
        w = 1.0
    else:
        w_star = c_min / c_max
        w = max(w_star, 0.5)

    # Calculate weighted average
    numerator = 0.0
    denominator = 0.0
    for pos, conc in zip(positions, concentrations):
        weight = w ** pos
        numerator += weight * conc
        denominator += weight

    if denominator == 0:
        return None

    return numerator / denominator


# ============================================================================
# Dominant Pollutant & Overall AQI
# ============================================================================

def calculate_individual_aqi(
    pm25: Optional[float] = None,
    pm10: Optional[float] = None,
    o3: Optional[float] = None,
    no2: Optional[float] = None,
    so2: Optional[float] = None,
    co: Optional[float] = None,
) -> Tuple[Optional[int], Optional[str], Dict[str, int]]:
    """Calculate individual AQI values and identify the dominant pollutant.

    The overall AQI is the maximum of the individual pollutant AQI values.

    Args:
        pm25: PM2.5 in ug/m3
        pm10: PM10 in ug/m3
        o3: O3 in ug/m3
        no2: NO2 in ug/m3
        so2: SO2 in ug/m3
        co: CO in ug/m3

    Returns:
        Tuple of (overall_aqi, dominant_pollutant_name, individual_aqi_dict)
    """
    individual = {}

    if pm25 is not None:
        aqi = calculate_pm25_aqi(pm25)
        if aqi is not None:
            individual["pm25"] = aqi

    if pm10 is not None:
        aqi = calculate_pm10_aqi(pm10)
        if aqi is not None:
            individual["pm10"] = aqi

    if o3 is not None:
        aqi = calculate_o3_aqi(o3)
        if aqi is not None:
            individual["o3"] = aqi

    if no2 is not None:
        aqi = calculate_no2_aqi(no2)
        if aqi is not None:
            individual["no2"] = aqi

    if so2 is not None:
        aqi = calculate_so2_aqi(so2)
        if aqi is not None:
            individual["so2"] = aqi

    if co is not None:
        aqi = calculate_co_aqi(co)
        if aqi is not None:
            individual["co"] = aqi

    if not individual:
        return None, None, {}

    dominant = max(individual, key=individual.get)
    overall_aqi = individual[dominant]

    return overall_aqi, dominant, individual


# ============================================================================
# NowCast-based AQI (Recommended Target Method)
# ============================================================================

def calculate_nowcast_aqi(
    pm25_hourly: List[Optional[float]],
    pm10_hourly: List[Optional[float]] = None,
) -> Tuple[Optional[int], Optional[str], Dict[str, Any]]:
    """Calculate EPA AQI using NowCast methodology for real-time hourly AQI.

    This is the recommended target definition for the AQI predictor project.
    It produces an AQI value at each hourly timestamp using the EPA NowCast
    algorithm, which is what AirNow uses for current conditions reporting.

    Methodology:
    1. Calculate NowCast concentration for PM2.5 (up to 12 hours)
    2. Calculate NowCast concentration for PM10 (up to 12 hours)
    3. Calculate individual pollutant AQIs
    4. Select the highest as the overall AQI

    Requirements:
    - At least 2 of past 3 hours must be valid
    - Most recent hour must be valid
    - For PM2.5: up to 12 hours of history
    - For PM10: up to 12 hours of history

    Args:
        pm25_hourly: List of PM2.5 hourly concentrations (most recent last)
        pm10_hourly: List of PM10 hourly concentrations (most recent last)

    Returns:
        Tuple of (overall_aqi, dominant_pollutant, metadata_dict)
    """
    # Calculate NowCast concentrations
    pm25_nowcast = calculate_nowcast(pm25_hourly)
    pm10_nowcast = calculate_nowcast(pm10_hourly) if pm10_hourly else None

    # Calculate individual AQIs from NowCast concentrations
    aqi, dominant, individual = calculate_individual_aqi(
        pm25=pm25_nowcast,
        pm10=pm10_nowcast,
    )

    metadata = {
        "method": "nowcast",
        "pm25_nowcast": pm25_nowcast,
        "pm10_nowcast": pm10_nowcast,
        "individual_aqi": individual,
        "dominant_pollutant": dominant,
        "hours_used_pm25": sum(1 for c in (pm25_hourly[-12:] if pm25_hourly else []) if c is not None),
        "hours_used_pm10": sum(1 for c in (pm10_hourly[-12:] if pm10_hourly else []) if c is not None),
    }

    return aqi, dominant, metadata


# ============================================================================
# Metadata
# ============================================================================

def get_aqi_metadata() -> Dict[str, str]:
    """Get AQI calculation metadata for audit trail.

    Returns:
        Dictionary with AQI standard, version, and reference.
    """
    return {
        "aqi_standard": AQI_STANDARD,
        "aqi_method_version": AQI_CALCULATION_VERSION,
        "aqi_derived": True,
        "aqi_source": "openweather_pollutants",
        "aqi_reference": AQI_CALCULATION_REFERENCE,
        "nowcast_reference": NOWCAST_REFERENCE,
        "pm25_breakpoint_note": "Updated May 2024: Good 0.0-9.0 ug/m3 (was 0.0-12.0)",
        "derived_disclosure": "US EPA-method AQI derived from OpenWeather pollutant concentrations. Not an official EPA/AirNow monitor reading.",
        "unit_conversion_note": "Gas concentrations (ug/m3) converted to ppm/ppb using standard conditions (25C, 1 atm). Error typically <5% for ambient conditions.",
    }
