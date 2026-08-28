"""
Tests for AQICN timestamp parsing — timezone-aware ISO-8601 handling.

Verifies correct UTC conversion for:
- Positive offsets (e.g. +05:00 Pakistan)
- Negative offsets (e.g. -05:00 US Eastern)
- UTC (+00:00)
- The exact AQICN payload formats observed in production
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.data.aqicn_client import _parse_aqicn_timestamp


class TestAQICTimezoneParsing:
    """Test correct timezone-aware timestamp parsing."""

    def test_pakistan_offset_positive(self):
        """Pakistan UTC+05:00 converts correctly to UTC."""
        time_data = {"iso": "2026-08-26T16:00:00+05:00", "v": 0, "tz": "+05:00"}
        result = _parse_aqicn_timestamp(time_data)
        assert result is not None
        assert result.tzinfo is not None
        # 16:00+05:00 = 11:00 UTC
        assert result.hour == 11
        assert result.minute == 0
        assert result.tzinfo == timezone.utc

    def test_us_eastern_offset_negative(self):
        """US Eastern UTC-05:00 converts correctly to UTC."""
        time_data = {"iso": "2026-08-26T11:00:00-05:00", "v": 0, "tz": "-05:00"}
        result = _parse_aqicn_timestamp(time_data)
        assert result is not None
        # 11:00-05:00 = 16:00 UTC
        assert result.hour == 16
        assert result.minute == 0

    def test_utc_offset_zero(self):
        """UTC+00:00 stays at the same time."""
        time_data = {"iso": "2026-08-26T12:00:00+00:00", "v": 0, "tz": "+00:00"}
        result = _parse_aqicn_timestamp(time_data)
        assert result is not None
        assert result.hour == 12
        assert result.minute == 0

    def test_utc_offset_negative_six(self):
        """US Central UTC-06:00 converts correctly to UTC."""
        time_data = {"iso": "2026-08-26T10:00:00-06:00", "v": 0, "tz": "-06:00"}
        result = _parse_aqicn_timestamp(time_data)
        assert result is not None
        # 10:00-06:00 = 16:00 UTC
        assert result.hour == 16
        assert result.minute == 0

    def test_naive_iso_treated_as_utc(self):
        """ISO string without offset is treated as UTC."""
        time_data = {"iso": "2026-08-26T12:00:00", "v": 0}
        result = _parse_aqicn_timestamp(time_data)
        assert result is not None
        assert result.hour == 12

    def test_unix_fallback_when_iso_missing(self):
        """Unix timestamp used only when ISO is unavailable."""
        time_data = {"v": 1787742000}
        result = _parse_aqicn_timestamp(time_data)
        assert result is not None
        # Unix 1787742000 = 2026-08-26T11:00:00 UTC
        assert result.hour == 11

    def test_iso_preferred_over_unix(self):
        """ISO string is preferred even when Unix is present."""
        # ISO says 16:00+05:00 = 11:00 UTC
        # Unix says 16:00 UTC (wrong — local time misinterpreted)
        time_data = {
            "iso": "2026-08-26T16:00:00+05:00",
            "v": 1787742000,  # This is 11:00 UTC, but AQICN means 16:00 local
            "tz": "+05:00",
        }
        result = _parse_aqicn_timestamp(time_data)
        assert result is not None
        # Should use ISO: 16:00+05:00 = 11:00 UTC
        assert result.hour == 11

    def test_exact_karachi_payload(self):
        """Exact AQICN payload observed for Karachi station @11790."""
        time_data = {
            "s": "2025-03-04 16:00:00",
            "tz": "+05:00",
            "v": 1741104000,
            "iso": "2025-03-04T16:00:00+05:00",
        }
        result = _parse_aqicn_timestamp(time_data)
        assert result is not None
        # 16:00+05:00 = 11:00 UTC
        assert result.year == 2025
        assert result.month == 3
        assert result.day == 4
        assert result.hour == 11
        assert result.minute == 0

    def test_exact_lahore_payload(self):
        """Exact AQICN payload observed for Lahore station @11765."""
        time_data = {
            "s": "2025-02-18 18:00:00",
            "tz": "+05:00",
            "v": 1739901600,
            "iso": "2025-02-18T18:00:00+05:00",
        }
        result = _parse_aqicn_timestamp(time_data)
        assert result is not None
        # 18:00+05:00 = 13:00 UTC
        assert result.year == 2025
        assert result.month == 2
        assert result.day == 18
        assert result.hour == 13

    def test_exact_islamabad_payload(self):
        """Exact AQICN payload observed for Islamabad station @11739."""
        time_data = {
            "s": "2026-02-16 17:00:00",
            "tz": "+05:00",
            "v": 1771261200,
            "iso": "2026-02-16T17:00:00+05:00",
        }
        result = _parse_aqicn_timestamp(time_data)
        assert result is not None
        # 17:00+05:00 = 12:00 UTC
        assert result.year == 2026
        assert result.month == 2
        assert result.day == 16
        assert result.hour == 12

    def test_none_time_data(self):
        """None time data returns None."""
        assert _parse_aqicn_timestamp(None) is None

    def test_empty_time_data(self):
        """Empty dict returns None."""
        assert _parse_aqicn_timestamp({}) is None

    def test_iso_only_no_unix(self):
        """ISO without Unix timestamp works correctly."""
        time_data = {"iso": "2026-08-26T16:00:00+05:00"}
        result = _parse_aqicn_timestamp(time_data)
        assert result is not None
        assert result.hour == 11  # 16:00+05:00 = 11:00 UTC
