"""
Real Data Collector

Collects real data from APIs with usage monitoring and audit trails.
"""

import os
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

import pandas as pd

from src.data.openweather_client import OpenWeatherClient
from src.data.aqicn_client import AQICNClient


@dataclass
class APICallRecord:
    """Record of an API call."""
    api_name: str
    endpoint: str
    timestamp: str
    status_code: int
    response_time_ms: float
    success: bool
    error_message: Optional[str] = None
    data_points: int = 0


@dataclass
class APIUsageMetrics:
    """API usage metrics."""
    api_name: str
    total_calls: int
    successful_calls: int
    failed_calls: int
    total_response_time_ms: float
    avg_response_time_ms: float
    data_points_collected: int
    rate_limit_hits: int
    first_call: str
    last_call: str


class RealDataCollector:
    """
    Collects real data from APIs with monitoring.
    
    Features:
    - API usage tracking
    - Rate limit monitoring
    - Audit trail
    - Resumable collection
    """
    
    # Valid cities
    CITIES = {
        "karachi": {"lat": 24.8607, "lon": 67.0011},
        "lahore": {"lat": 31.5204, "lon": 74.3587},
        "islamabad": {"lat": 33.6844, "lon": 73.0479},
    }
    
    def __init__(
        self,
        openweather_key: Optional[str] = None,
        aqicn_key: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize real data collector.
        
        Args:
            openweather_key: OpenWeather API key
            aqicn_key: AQICN API key
            output_dir: Output directory for collected data
        """
        self.openweather_key = openweather_key or os.getenv("OPENWEATHER_API_KEY")
        self.aqicn_key = aqicn_key or os.getenv("AQICN_API_KEY")
        
        self.output_dir = output_dir or Path("data/raw/real")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize API clients
        self.openweather_client = None
        if self.openweather_key:
            self.openweather_client = OpenWeatherClient(api_key=self.openweather_key)
        
        self.aqicn_client = None
        if self.aqicn_key:
            self.aqicn_client = AQICNClient(api_key=self.aqicn_key)
        
        # Usage tracking
        self.call_records: List[APICallRecord] = []
        self._load_existing_records()
    
    def _load_existing_records(self):
        """Load existing API call records."""
        records_file = self.output_dir / "api_call_records.json"
        if records_file.exists():
            with open(records_file, "r") as f:
                records = json.load(f)
                self.call_records = [APICallRecord(**r) for r in records]
    
    def _save_records(self):
        """Save API call records."""
        records_file = self.output_dir / "api_call_records.json"
        with open(records_file, "w") as f:
            json.dump([asdict(r) for r in self.call_records], f, indent=2)
    
    def _record_call(
        self,
        api_name: str,
        endpoint: str,
        status_code: int,
        response_time_ms: float,
        success: bool,
        error_message: Optional[str] = None,
        data_points: int = 0,
    ):
        """Record an API call."""
        record = APICallRecord(
            api_name=api_name,
            endpoint=endpoint,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status_code=status_code,
            response_time_ms=response_time_ms,
            success=success,
            error_message=error_message,
            data_points=data_points,
        )
        self.call_records.append(record)
        self._save_records()
    
    def collect_current_data(self, city: str) -> Dict[str, Any]:
        """
        Collect current data for a city.
        
        Args:
            city: City name
            
        Returns:
            Collected data
        """
        city_lower = city.lower()
        if city_lower not in self.CITIES:
            raise ValueError(f"Invalid city: {city}")
        
        coords = self.CITIES[city_lower]
        collected_data = {}
        
        # Collect from OpenWeather
        if self.openweather_client:
            start_time = time.time()
            try:
                weather_data = self.openweather_client.get_current_weather(
                    lat=coords["lat"],
                    lon=coords["lon"],
                )
                response_time = (time.time() - start_time) * 1000
                
                self._record_call(
                    api_name="openweather",
                    endpoint="current_weather",
                    status_code=200,
                    response_time_ms=response_time,
                    success=True,
                    data_points=1,
                )
                
                collected_data["openweather_weather"] = weather_data
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                self._record_call(
                    api_name="openweather",
                    endpoint="current_weather",
                    status_code=0,
                    response_time_ms=response_time,
                    success=False,
                    error_message=str(e),
                )
        
        # Collect from AQICN
        if self.aqicn_client:
            start_time = time.time()
            try:
                aqicn_data = self.aqicn_client.get_station_data(city_lower)
                response_time = (time.time() - start_time) * 1000
                
                self._record_call(
                    api_name="aqicn",
                    endpoint="station_data",
                    status_code=200,
                    response_time_ms=response_time,
                    success=True,
                    data_points=1,
                )
                
                collected_data["aqicn_data"] = aqicn_data
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                self._record_call(
                    api_name="aqicn",
                    endpoint="station_data",
                    status_code=0,
                    response_time_ms=response_time,
                    success=False,
                    error_message=str(e),
                )
        
        return collected_data
    
    def collect_all_cities(self) -> Dict[str, Dict[str, Any]]:
        """
        Collect data for all cities.
        
        Returns:
            Collected data for all cities
        """
        all_data = {}
        
        for city in self.CITIES.keys():
            print(f"Collecting data for {city}...")
            all_data[city] = self.collect_current_data(city)
            time.sleep(1)  # Rate limiting
        
        return all_data
    
    def get_usage_metrics(self) -> Dict[str, APIUsageMetrics]:
        """
        Get API usage metrics.
        
        Returns:
            Usage metrics per API
        """
        metrics = {}
        
        # Group records by API
        api_records = {}
        for record in self.call_records:
            if record.api_name not in api_records:
                api_records[record.api_name] = []
            api_records[record.api_name].append(record)
        
        # Calculate metrics per API
        for api_name, records in api_records.items():
            successful = [r for r in records if r.success]
            failed = [r for r in records if not r.success]
            rate_limit_hits = sum(1 for r in records if r.status_code == 429)
            
            metrics[api_name] = APIUsageMetrics(
                api_name=api_name,
                total_calls=len(records),
                successful_calls=len(successful),
                failed_calls=len(failed),
                total_response_time_ms=sum(r.response_time_ms for r in records),
                avg_response_time_ms=(
                    sum(r.response_time_ms for r in records) / len(records)
                    if records else 0
                ),
                data_points_collected=sum(r.data_points for r in records),
                rate_limit_hits=rate_limit_hits,
                first_call=records[0].timestamp if records else "",
                last_call=records[-1].timestamp if records else "",
            )
        
        return metrics
    
    def save_collected_data(
        self,
        data: Dict[str, Any],
        city: str,
    ) -> Path:
        """
        Save collected data.
        
        Args:
            data: Collected data
            city: City name
            
        Returns:
            Path to saved file
        """
        city_dir = self.output_dir / city
        city_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = city_dir / f"collection_{timestamp}.json"
        
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        return output_file
    
    def get_collection_summary(self) -> Dict[str, Any]:
        """
        Get collection summary.
        
        Returns:
            Collection summary
        """
        metrics = self.get_usage_metrics()
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_calls": len(self.call_records),
            "apis": {
                name: {
                    "total_calls": m.total_calls,
                    "successful_calls": m.successful_calls,
                    "failed_calls": m.failed_calls,
                    "data_points": m.data_points_collected,
                    "rate_limit_hits": m.rate_limit_hits,
                }
                for name, m in metrics.items()
            },
        }


def main():
    """Main collection entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect real data from APIs")
    parser.add_argument("--city", help="Specific city to collect (all if not specified)")
    parser.add_argument("--save", action="store_true", help="Save collected data")
    
    args = parser.parse_args()
    
    collector = RealDataCollector()
    
    if args.city:
        data = collector.collect_current_data(args.city)
        print(f"Collected data for {args.city}")
        if args.save:
            collector.save_collected_data(data, args.city)
    else:
        data = collector.collect_all_cities()
        print("Collected data for all cities")
        if args.save:
            for city, city_data in data.items():
                collector.save_collected_data(city_data, city)
    
    # Print summary
    summary = collector.get_collection_summary()
    print("\n" + "=" * 60)
    print("Collection Summary")
    print("=" * 60)
    print(f"Total API calls: {summary['total_calls']}")
    for api_name, api_stats in summary["apis"].items():
        print(f"{api_name}: {api_stats['successful_calls']} successful, "
              f"{api_stats['failed_calls']} failed")
    print("=" * 60)


if __name__ == "__main__":
    main()
