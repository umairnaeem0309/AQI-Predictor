"""
US EPA AQI Calculation from Pollutant Concentrations.

Converts hourly pollutant concentrations (from OpenWeather) to US EPA AQI (0-500 scale).
Uses current EPA breakpoints effective May 6, 2024 (PM2.5 annual standard updated to 9 ug/m3).

References:
- EPA AQI Breakpoints: https://aqs.epa.gov/aqsweb/documents/codetables/aqi_breakpoints.html
- EPA NowCast Methodology: https://www.airnow.gov/aqi/the-aqi-equation/
- EPA PM2.5 Update (May 2024): 9 ug/m3 annual standard

NOT for production use without validation against known EPA examples.
"""
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

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

# O3 (ppm, 8-hour average) — Note: OpenWeather returns O3 in ug/m3
# EPA breakpoints are in ppm for 8-hour O3
# Conversion: ug/m3 = ppm * (M / 24.45) where M=48 for O3
# So ppm = ug/m3 / (48 / 24.45) = ug/m3 / 1.963
O3_BREAKPOINTS_PPM: List[BreakpointRow] = [
    BreakpointRow("o3", 0, 50, 0.0, 0.054, "ppm", "8-hour"),
    BreakpointRow("o3", 51, 100, 0.055, 0.070, "ppm", "8-hour"),
    BreakpointRow("o3", 101, 150, 0.071, 0.085, "ppm", "8-hour"),
    BreakpointRow("o3", 151, 200, 0.086, 0.105, "ppm", "8-hour"),
    BreakpointRow("o3", 201, 300, 0.106, 0.200, "ppm", "8-hour"),
]

# NO2 (ppb, 1-hour) — Note: OpenWeather returns NO2 in ug/m3
# EPA breakpoints are in ppb for 1-hour NO2
# Conversion: ug/m3 = ppb * (M / 24.45) where M=46 for NO2
# So ppb = ug/m3 / (46 / 24.45) = ug/m3 / 1.881
NO2_BREAKPOINTS_PPB: List[BreakpointRow] = [
    BreakpointRow("no2", 0, 50, 0.0, 53.0, "ppb", "1-hour"),
    BreakpointRow("no2", 51, 100, 54.0, 100.0, "ppb", "1-hour"),
    BreakpointRow("no2", 101, 150, 101.0, 360.0, "ppb", "1-hour"),
    BreakpointRow("no2", 151, 200, 361.0, 649.0, "ppb", "1-hour"),
    BreakpointRow("no2", 201, 300, 650.0, 1249.0, "ppb", "1-hour"),
    BreakpointRow("no2", 301, 400, 1250.0, 2049.0, "ppb", "1-hour"),
    BreakpointRow("no2", 401, 500, 2050.0, 3049.0, "ppb", "1-hour"),
]

# SO2 (ppb, 1-hour) — Note: OpenWeather returns SO2 in ug/m3
# EPA breakpoints are in ppb for 1-hour SO2
# Conversion: ug/m3 = ppb * (M / 24.45) where M=64 for SO2
# So ppb = ug/m3 / (64 / 24.45) = ug/m3 / 2.618
SO2_BREAKPOINTS_PPB: List[BreakpointRow] = [
    BreakpointRow("so2", 0, 50, 0.0, 35.0, "ppb", "1-hour"),
    BreakpointRow("so2", 51, 100, 36.0, 75.0, "ppb", "1-hour"),
    BreakpointRow("so2", 101, 150, 76.0, 185.0, "ppb", "1-hour"),
    BreakpointRow("so2", 151, 200, 186.0, 304.0, "ppb", "1-hour"),
    BreakpointRow("so2", 201, 300, 305.0, 604.0, "ppb", "1-hour"),
    BreakpointRow("so2", 301, 400, 605.0, 1004.0, "ppb", "1-hour"),
    BreakpointRow("so2", 401, 500, 1005.0, 2004.0, "ppb", "1-hour"),
]

# CO (ppm, 8-hour) — Note: OpenWeather returns CO in ug/m3
# EPA breakpoints are in ppm for 8-hour CO
# Conversion: ug/m3 = ppm * (M / 24.45) where M=28 for CO
# So ppm = ug/m3 / (28 / 24.45) = ug/m3 / 1.145
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


# ============================================================================
# Unit Conversion Functions
# ============================================================================

def ug_m3_to_ppm(conc_ug_m3: float, molecular_weight: float) -> float:
    """Convert ug/m3 to ppm at standard conditions (25°C, 1 atm).

    At standard conditions (25°C, 1 atm):
        Molar volume = 24.45 L/mol = 0.02445 m3/mol
        Concentration (ug/m3) = (ppm * M * 1000) / 24.45
        Therefore: ppm = (conc_ug_m3 * 24.45) / (M * 1000)

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

    Args:
        conc_ug_m3: Concentration in ug/m3
        molecular_weight: Molecular weight of the gas (g/mol)

    Returns:
        Concentration in ppb
    """
    return (conc_ug_m3 * 24.45) / molecular_weight


# ============================================================================
# AQI Calculation
# ============================================================================

def calculate_aqi_from_concentration(
    concentration: float,
    breakpoints: List[BreakpointRow],
) -> Optional[int]:
    """Calculate AQI from concentration using breakpoint table.

    Uses the standard EPA linear interpolation formula:
        AQI = ((AQI_high - AQI_low) / (C_high - C_low)) * (C - C_low) + AQI_low

    Args:
        concentration: Pollutant concentration in the correct unit
        breakpoints: List of breakpoint rows for this pollutant

    Returns:
        AQI value (integer, rounded to nearest whole number) or None if out of range
    """
    if concentration is None or math.isnan(concentration):
        return None

    for bp in breakpoints:
        if bp.conc_low <= concentration <= bp.conc_high:
            # Linear interpolation
            aqi_range = bp.aqi_high - bp.aqi_low
            conc_range = bp.conc_high - bp.conc_low
            if conc_range == 0:
                return bp.aqi_low
            aqi = ((aqi_range / conc_range) * (concentration - bp.conc_low)) + bp.aqi_low
            return round(aqi)

    # Beyond breakpoint range
    if concentration > breakpoints[-1].conc_high:
        return 500  # Cap at hazardous
    return None  # Below lowest breakpoint


def calculate_pm25_aqi(pm25_ug_m3: float) -> Optional[int]:
    """Calculate AQI from PM2.5 concentration (ug/m3).

    Uses 2024 EPA breakpoints (Good: 0.0-9.0 ug/m3).

    Args:
        pm25_ug_m3: PM2.5 concentration in ug/m3

    Returns:
        AQI value (0-500) or None
    """
    return calculate_aqi_from_concentration(pm25_ug_m3, PM25_BREAKPOINTS)


def calculate_pm10_aqi(pm10_ug_m3: float) -> Optional[int]:
    """Calculate AQI from PM10 concentration (ug/m3).

    Args:
        pm10_ug_m3: PM10 concentration in ug/m3

    Returns:
        AQI value (0-500) or None
    """
    return calculate_aqi_from_concentration(pm10_ug_m3, PM10_BREAKPOINTS)


def calculate_o3_aqi(o3_ug_m3: float) -> Optional[int]:
    """Calculate AQI from O3 concentration (ug/m3).

    Converts ug/m3 to ppm, then applies EPA 8-hour O3 breakpoints.

    Args:
        o3_ug_m3: O3 concentration in ug/m3

    Returns:
        AQI value (0-300 for 8-hour) or None
    """
    if o3_ug_m3 is None:
        return None
    o3_ppm = ug_m3_to_ppm(o3_ug_m3, 48.0)  # O3 molecular weight = 48
    return calculate_aqi_from_concentration(o3_ppm, O3_BREAKPOINTS_PPM)


def calculate_no2_aqi(no2_ug_m3: float) -> Optional[int]:
    """Calculate AQI from NO2 concentration (ug/m3).

    Converts ug/m3 to ppb, then applies EPA 1-hour NO2 breakpoints.

    Args:
        no2_ug_m3: NO2 concentration in ug/m3

    Returns:
        AQI value (0-500) or None
    """
    if no2_ug_m3 is None:
        return None
    no2_ppb = ug_m3_to_ppb(no2_ug_m3, 46.0)  # NO2 molecular weight = 46
    return calculate_aqi_from_concentration(no2_ppb, NO2_BREAKPOINTS_PPB)


def calculate_so2_aqi(so2_ug_m3: float) -> Optional[int]:
    """Calculate AQI from SO2 concentration (ug/m3).

    Converts ug/m3 to ppb, then applies EPA 1-hour SO2 breakpoints.

    Args:
        so2_ug_m3: SO2 concentration in ug/m3

    Returns:
        AQI value (0-500) or None
    """
    if so2_ug_m3 is None:
        return None
    so2_ppb = ug_m3_to_ppb(so2_ug_m3, 64.0)  # SO2 molecular weight = 64
    return calculate_aqi_from_concentration(so2_ppb, SO2_BREAKPOINTS_PPB)


def calculate_co_aqi(co_ug_m3: float) -> Optional[int]:
    """Calculate AQI from CO concentration (ug/m3).

    Converts ug/m3 to ppm, then applies EPA 8-hour CO breakpoints.

    Args:
        co_ug_m3: CO concentration in ug/m3

    Returns:
        AQI value (0-500) or None
    """
    if co_ug_m3 is None:
        return None
    co_ppm = ug_m3_to_ppm(co_ug_m3, 28.0)  # CO molecular weight = 28
    return calculate_aqi_from_concentration(co_ppm, CO_BREAKPOINTS_PPM)


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
) -> Tuple[Optional[int], Optional[str]]:
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
        Tuple of (overall_aqi, dominant_pollutant_name) or (None, None)
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
        return None, None

    # Overall AQI = max of individual AQI values
    dominant = max(individual, key=individual.get)
    overall_aqi = individual[dominant]

    return overall_aqi, dominant


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
        "pm25_breakpoint_note": "Updated May 2024: Good 0.0-9.0 ug/m3 (was 0.0-12.0)",
    }
