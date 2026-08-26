"""
Real Data Collector

Collects real data from APIs using the APIManager orchestrator.
Tracks API usage metrics and audit trails.
"""

import os
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

import pandas as pd

from src.config import load_environment, get_api_key
from src.data.api_manager import APIManager
from src.data.schemas import CityConfig

logger = logging.getLogger(__name__)


@dataclass
class CollectionRound:
    """Record of one collection round across all cities."""
    round_id: str
    timestamp: str
    cities_attempted: int
    cities_succeeded: int
    cities_failed: List[str]
    observations_count: int
    collection_duration_ms: float
    source_summary: Dict[str, int]


class RealDataCollector:
    """
    Collects real data from APIs with monitoring.

    Delegates actual API calls to APIManager (which orchestrates
    OpenWeather and AQICN). This collector adds:
    - Collection round tracking
    - Audit trail (saved observations)
    - Resumable collection support

    Features:
    - Audit trail
    - Resumable collection (appends to master CSV)
    """

    CITIES = {
        "karachi": CityConfig(id="karachi", name="Karachi", latitude=24.8607, longitude=67.0011),
        "lahore": CityConfig(id="lahore", name="Lahore", latitude=31.5204, longitude=74.3587),
        "islamabad": CityConfig(id="islamabad", name="Islamabad", latitude=33.6844, longitude=73.0479),
    }

    def __init__(
        self,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize real data collector.

        Args:
            output_dir: Output directory for collected data.
                       Defaults to data/raw/real
        """
        self.output_dir = output_dir or Path("data/raw/real")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize NowCast history manager
        from src.data.nowcast_history import NowCastHistoryManager
        self.nowcast_history = NowCastHistoryManager(
            history_path=self.output_dir / "nowcast_history.json"
        )

        # Load warm-up data into NowCast history if available
        master_csv = self.output_dir / "master_observations.csv"
        if master_csv.exists():
            self.nowcast_history.load_from_master_csv(master_csv)
            for city_id in ["karachi", "lahore", "islamabad"]:
                count = self.nowcast_history.get_history_count(city_id)
                if count > 0:
                    logger.info(
                        "Loaded %d NowCast history entries for %s from CSV",
                        count, city_id,
                    )

        # Initialize APIManager with NowCast history
        self.api_manager = APIManager(nowcast_history=self.nowcast_history)

        # Collection rounds tracking
        self.rounds: List[CollectionRound] = []
        self._load_existing_rounds()

    def _load_existing_rounds(self):
        """Load existing collection round records."""
        rounds_file = self.output_dir / "collection_rounds.json"
        if rounds_file.exists():
            with open(rounds_file, "r") as f:
                records = json.load(f)
                self.rounds = [CollectionRound(**r) for r in records]

    def _save_rounds(self):
        """Save collection round records."""
        rounds_file = self.output_dir / "collection_rounds.json"
        with open(rounds_file, "w") as f:
            json.dump([asdict(r) for r in self.rounds], f, indent=2)

    def collect_round(
        self,
        cities: Optional[List[str]] = None,
        save_raw: bool = True,
    ) -> pd.DataFrame:
        """
        Perform one collection round for specified cities.

        Args:
            cities: List of city IDs to collect. If None, collects all.
            save_raw: Whether to save raw data to audit directory.

        Returns:
            DataFrame with collected observations.
        """
        start_time = time.time()
        round_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Determine which cities to collect
        if cities is None:
            city_configs = list(self.CITIES.values())
        else:
            city_configs = [
                self.CITIES[c.lower()]
                for c in cities
                if c.lower() in self.CITIES
            ]

        if not city_configs:
            raise ValueError(f"No valid cities in: {cities}")

        # Collect via APIManager
        df = self.api_manager.fetch_all_cities(city_configs=city_configs)
        collection_duration_ms = (time.time() - start_time) * 1000

        # Track which cities succeeded/failed
        collected_cities = set()
        failed_cities = []
        if not df.empty and "location_id" in df.columns:
            collected_cities = set(df["location_id"].unique())
        failed_cities = [
            c.id for c in city_configs
            if c.id not in collected_cities
        ]

        # Source summary
        source_summary = {}
        if not df.empty:
            for col in ["weather_source", "pollution_source"]:
                if col in df.columns:
                    source_summary[col] = df[col].value_counts().to_dict()

        # Record this collection round
        collection_round = CollectionRound(
            round_id=round_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            cities_attempted=len(city_configs),
            cities_succeeded=len(collected_cities),
            cities_failed=failed_cities,
            observations_count=len(df),
            collection_duration_ms=collection_duration_ms,
            source_summary=source_summary,
        )
        self.rounds.append(collection_round)
        self._save_rounds()

        # Persist NowCast history for next round
        self.nowcast_history.save_history()

        # Save raw data
        if save_raw and not df.empty:
            self._save_round_data(df, round_id)

        return df

    def _save_round_data(self, df: pd.DataFrame, round_id: str):
        """Save round data to audit directory and append to master CSV."""
        # Save raw audit JSON for this round
        audit_dir = self.output_dir / "rounds"
        audit_dir.mkdir(parents=True, exist_ok=True)
        round_file = audit_dir / f"round_{round_id}.csv"
        df.to_csv(round_file, index=False)
        print(f"Round data saved: {round_file}")

        # Append to master CSV
        master_file = self.output_dir / "master_observations.csv"
        if master_file.exists():
            existing_df = pd.read_csv(master_file)
            combined = pd.concat([existing_df, df], ignore_index=True)
            # Remove duplicates by (timestamp, location_id)
            if "timestamp" in combined.columns and "location_id" in combined.columns:
                combined = combined.drop_duplicates(
                    subset=["timestamp", "location_id"], keep="last"
                )
            combined.to_csv(master_file, index=False)
        else:
            df.to_csv(master_file, index=False)
        print(f"Master CSV updated: {master_file}")

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get collection usage summary across all rounds."""
        if not self.rounds:
            return {"message": "No collection rounds recorded"}

        total_cities = sum(r.cities_attempted for r in self.rounds)
        total_succeeded = sum(r.cities_succeeded for r in self.rounds)
        total_observations = sum(r.observations_count for r in self.rounds)
        total_duration = sum(r.collection_duration_ms for r in self.rounds)
        failed_cities = []
        for r in self.rounds:
            failed_cities.extend(r.cities_failed)

        first_round = self.rounds[0]
        last_round = self.rounds[-1]

        return {
            "total_rounds": len(self.rounds),
            "total_cities_attempted": total_cities,
            "total_cities_succeeded": total_succeeded,
            "total_observations": total_observations,
            "avg_duration_ms": total_duration / len(self.rounds),
            "total_duration_ms": total_duration,
            "unique_failed_cities": list(set(failed_cities)),
            "first_round": first_round.timestamp,
            "last_round": last_round.timestamp,
        }

    def get_master_dataframe(self) -> pd.DataFrame:
        """Load the master observations CSV."""
        master_file = self.output_dir / "master_observations.csv"
        if master_file.exists():
            return pd.read_csv(master_file)
        return pd.DataFrame()


def main():
    """Main collection entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Collect real data from APIs")
    parser.add_argument("--city", help="Specific city to collect (all if not specified)")
    parser.add_argument("--no-save", action="store_true", help="Skip saving data")
    parser.add_argument("--summary", action="store_true", help="Print usage summary")

    args = parser.parse_args()

    # Load environment
    load_environment()

    collector = RealDataCollector()

    if args.summary:
        summary = collector.get_usage_summary()
        print("\n" + "=" * 60)
        print("Collection Usage Summary")
        print("=" * 60)
        for key, value in summary.items():
            print(f"  {key}: {value}")
        print("=" * 60)
        return

    cities = [args.city] if args.city else None
    df = collector.collect_round(cities=cities, save_raw=not args.no_save)

    # Print results
    print("\n" + "=" * 60)
    print("Collection Summary")
    print("=" * 60)
    if df.empty:
        print("  No observations collected")
    else:
        print(f"  Observations: {len(df)}")
        if "location_id" in df.columns:
            print(f"  Cities: {df['location_id'].nunique()}")
            for city in df["location_id"].unique():
                city_count = len(df[df["location_id"] == city])
                print(f"    {city}: {city_count} observations")
    print("=" * 60)


if __name__ == "__main__":
    main()
