"""
NowCast History Manager — Maintains persistent hourly PM2.5/PM10 history per city.

The EPA NowCast algorithm requires up to 12 hours of hourly pollutant
concentrations to compute the NowCast concentration. This manager:

- Loads historical pollution data (warm-up + forward collection)
- Maintains a per-city sliding window of up to 12 hours
- Persists the history between collection rounds
- Provides the history to the NowCast AQI calculator

Storage: data/raw/real/nowcast_history.json
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Maximum NowCast window size
MAX_HISTORY_HOURS = 12


class NowCastHistoryManager:
    """Manages persistent per-city hourly PM2.5/PM10 history for NowCast.

    History is stored as a JSON file with structure:
    {
        "karachi": [
            {"timestamp": "2026-08-26T10:00:00Z", "pm25": 45.2, "pm10": 78.1},
            ...
        ],
        ...
    }

    Only the last MAX_HISTORY_HOURS entries per city are retained.
    """

    def __init__(self, history_path: Optional[Path] = None):
        """Initialize the NowCast history manager.

        Args:
            history_path: Path to the history JSON file.
                         Defaults to data/raw/real/nowcast_history.json
        """
        self.history_path = history_path or Path("data/raw/real/nowcast_history.json")
        self.history: Dict[str, List[dict]] = self._load_history()

    def _load_history(self) -> Dict[str, List[dict]]:
        """Load history from disk."""
        if self.history_path.exists():
            try:
                with open(self.history_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Failed to load NowCast history: %s", e)
                return {}
        return {}

    def save_history(self) -> None:
        """Persist current history to disk."""
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.debug("NowCast history saved: %d cities", len(self.history))

    def add_observation(
        self,
        city_id: str,
        timestamp: str,
        pm25: Optional[float],
        pm10: Optional[float],
    ) -> None:
        """Add a new hourly observation to the history for a city.

        Args:
            city_id: City identifier (e.g. "karachi")
            timestamp: ISO-8601 UTC timestamp
            pm25: PM2.5 concentration in ug/m3 (or None)
            pm10: PM10 concentration in ug/m3 (or None)
        """
        if city_id not in self.history:
            self.history[city_id] = []

        entry = {
            "timestamp": timestamp,
            "pm25": pm25,
            "pm10": pm10,
        }
        self.history[city_id].append(entry)

        # Sort by timestamp and deduplicate
        self.history[city_id].sort(key=lambda x: x["timestamp"])
        seen = set()
        deduped = []
        for e in self.history[city_id]:
            if e["timestamp"] not in seen:
                seen.add(e["timestamp"])
                deduped.append(e)
        self.history[city_id] = deduped

        # Keep only the last MAX_HISTORY_HOURS entries
        self.history[city_id] = self.history[city_id][-MAX_HISTORY_HOURS:]

    def add_warmup_data(
        self,
        city_id: str,
        warmup_df,
    ) -> int:
        """Add historical warm-up pollution data for a city.

        Args:
            city_id: City identifier
            warmup_df: DataFrame with columns: timestamp, pm25, pm10

        Returns:
            Number of entries added
        """
        count = 0
        for _, row in warmup_df.iterrows():
            ts = str(row["timestamp"])
            pm25 = float(row["pm25"]) if row.get("pm25") is not None else None
            pm10 = float(row["pm10"]) if row.get("pm10") is not None else None
            self.add_observation(city_id, ts, pm25, pm10)
            count += 1

        logger.info("Added %d warm-up entries for %s", count, city_id)
        return count

    def get_history(
        self,
        city_id: str,
        max_hours: int = MAX_HISTORY_HOURS,
    ) -> Tuple[List[Optional[float]], List[Optional[float]], List[str]]:
        """Get the PM2.5 and PM10 history for NowCast calculation.

        Args:
            city_id: City identifier
            max_hours: Maximum hours of history to return

        Returns:
            Tuple of (pm25_hourly, pm10_hourly, timestamps)
            where each list is ordered oldest-first, most recent last.
        """
        entries = self.history.get(city_id, [])[-max_hours:]

        pm25_hourly = [e.get("pm25") for e in entries]
        pm10_hourly = [e.get("pm10") for e in entries]
        timestamps = [e.get("timestamp") for e in entries]

        return pm25_hourly, pm10_hourly, timestamps

    def get_history_count(self, city_id: str) -> int:
        """Get the number of hours of history for a city."""
        return len(self.history.get(city_id, []))

    def load_from_master_csv(
        self,
        csv_path: Path,
        city_id: Optional[str] = None,
    ) -> Dict[str, int]:
        """Load history from master observations CSV.

        Useful for initializing from warm-up data collected previously.

        Args:
            csv_path: Path to master_observations.csv
            city_id: If provided, only load data for this city

        Returns:
            Dict of city_id -> number of entries loaded
        """
        import pandas as pd

        if not csv_path.exists():
            logger.warning("Master CSV not found: %s", csv_path)
            return {}

        df = pd.read_csv(csv_path)

        # Filter to warmup pollution data (which has pm25/pm10 but no weather)
        if "data_type" in df.columns:
            warmup = df[df["data_type"] == "warmup_pollution"]
        else:
            # Fallback: use rows with pm25 data but no temperature
            warmup = df[df["pm25"].notna()]

        if city_id:
            warmup = warmup[warmup["location_id"] == city_id]

        counts = {}
        for cid in warmup["location_id"].unique():
            city_data = warmup[warmup["location_id"] == cid]
            count = 0
            for _, row in city_data.iterrows():
                ts = str(row["timestamp"])
                pm25 = float(row["pm25"]) if pd.notna(row.get("pm25")) else None
                pm10 = float(row["pm10"]) if pd.notna(row.get("pm10")) else None
                self.add_observation(cid, ts, pm25, pm10)
                count += 1
            counts[cid] = count
            logger.info("Loaded %d entries from CSV for %s", count, cid)

        return counts

    def clear_city(self, city_id: str) -> None:
        """Clear history for a specific city."""
        if city_id in self.history:
            del self.history[city_id]

    def clear_all(self) -> None:
        """Clear all history."""
        self.history = {}
