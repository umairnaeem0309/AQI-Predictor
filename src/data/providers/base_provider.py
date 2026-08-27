"""
Base Historical Provider — Abstract interface for bulk data providers.

Responsibilities:
- Define the contract for historical data retrieval
- Enforce chunked download for large date ranges
- Handle rate limiting and retry logic
- Provide standardized DataFrame output

Each concrete provider implements:
- _fetch_chunk(): Download one chunk of data from the API
- _parse_response(): Transform raw API response to DataFrame
- _get_variable_mapping(): Map internal column names to API variable names
"""

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class BaseHistoricalProvider(ABC):
    """Abstract base class for historical data providers.

    Subclasses must implement:
    - _fetch_chunk(): Download a chunk of hourly data
    - _parse_response(): Transform API response to DataFrame
    - _get_variable_mapping(): Map internal names to API variable names

    Attributes:
        base_url: API base URL.
        max_days_per_request: Maximum days per API request (API constraint).
        request_delay: Delay between requests in seconds (rate limiting).
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts for transient errors.
    """

    # Subclasses override these
    base_url: str = ""
    max_days_per_request: int = 90
    request_delay: float = 0.5
    timeout: int = 60
    max_retries: int = 3

    def __init__(
        self,
        timeout: int = 60,
        max_retries: int = 3,
        request_delay: float = 0.5,
    ):
        """Initialize the provider.

        Args:
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts for transient errors.
            request_delay: Delay between requests in seconds (rate limiting).
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.request_delay = request_delay
        self._session = requests.Session()
        self._request_count = 0
        self._error_count = 0

    @abstractmethod
    def _fetch_chunk(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Download one chunk of data from the API.

        Args:
            latitude: Location latitude.
            longitude: Location longitude.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            **kwargs: Provider-specific parameters.

        Returns:
            Raw API response as dictionary.

        Raises:
            requests.RequestException: On HTTP errors after retries.
        """
        pass

    @abstractmethod
    def _parse_response(
        self,
        raw_json: Dict[str, Any],
        location_id: str,
        city_name: str,
    ) -> pd.DataFrame:
        """Transform raw API response to standardized DataFrame.

        Must produce columns:
        - timestamp (datetime, UTC)
        - location_id (str)
        - city_name (str)
        - Plus provider-specific variable columns

        Args:
            raw_json: Raw API response dictionary.
            location_id: City identifier.
            city_name: Human-readable city name.

        Returns:
            DataFrame with standardized columns.
        """
        pass

    @abstractmethod
    def _get_variable_mapping(self) -> Dict[str, str]:
        """Return mapping of internal column names to API variable names.

        Returns:
            Dict mapping internal name (e.g., 'temperature') to API name
            (e.g., 'temperature_2m').
        """
        pass

    def _make_request(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make an HTTP GET request with retry logic.

        Retries on transient errors (network, timeout, 429, 5xx).
        Does not retry on client errors (4xx except 429).

        Args:
            url: Full request URL.
            params: Query parameters.

        Returns:
            Parsed JSON response.

        Raises:
            requests.RequestException: After all retries exhausted.
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                response = self._session.get(url, params=params, timeout=self.timeout)
                self._request_count += 1

                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(
                        "Rate limited (429). Waiting %d seconds...", retry_after
                    )
                    time.sleep(retry_after)
                    continue

                if response.status_code >= 500:
                    raise requests.RequestException(
                        f"Server error: HTTP {response.status_code}"
                    )

                if response.status_code == 400:
                    error_body = response.json() if response.content else {}
                    raise ValueError(
                        f"Bad request (400): {error_body.get('reason', response.text)}"
                    )

                response.raise_for_status()
                return response.json()

            except (requests.ConnectionError, requests.Timeout) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = 2 ** attempt
                    logger.warning(
                        "Request attempt %d/%d failed: %s. Retrying in %ds...",
                        attempt + 1,
                        self.max_retries, str(e), delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "All %d retry attempts exhausted. Last error: %s",
                        self.max_retries, str(e),
                    )

        raise last_exception

    def fetch_historical(
        self,
        latitude: float,
        longitude: float,
        location_id: str,
        city_name: str,
        start_date: str,
        end_date: str,
        **kwargs,
    ) -> pd.DataFrame:
        """Download historical data for a location over a date range.

        Automatically chunks large requests to respect API limits.
        Each chunk is fetched, parsed, and concatenated.

        Args:
            latitude: Location latitude.
            longitude: Location longitude.
            location_id: City identifier.
            city_name: Human-readable city name.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            **kwargs: Provider-specific parameters.

        Returns:
            DataFrame with hourly observations for the location.
        """
        logger.info(
            "Fetching %s historical data for %s (%.2f, %.2f) from %s to %s",
            self.__class__.__name__, city_name, latitude, longitude,
            start_date, end_date,
        )

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        chunks = []
        current_start = start

        while current_start <= end:
            # Calculate chunk end date (respect API limit)
            chunk_end = min(
                current_start + timedelta(days=self.max_days_per_request - 1),
                end,
            )

            chunk_start_str = current_start.strftime("%Y-%m-%d")
            chunk_end_str = chunk_end.strftime("%Y-%m-%d")

            logger.debug(
                "  Fetching chunk: %s to %s", chunk_start_str, chunk_end_str
            )

            try:
                raw_json = self._fetch_chunk(
                    latitude, longitude,
                    chunk_start_str, chunk_end_str,
                    **kwargs,
                )
                df_chunk = self._parse_response(raw_json, location_id, city_name)
                if not df_chunk.empty:
                    chunks.append(df_chunk)
                    logger.debug("  Chunk returned %d rows", len(df_chunk))

            except Exception as e:
                logger.error(
                    "  Failed to fetch chunk %s to %s: %s",
                    chunk_start_str, chunk_end_str, str(e),
                )
                self._error_count += 1

            # Move to next chunk
            current_start = chunk_end + timedelta(days=1)

            # Rate limiting delay
            if current_start <= end:
                time.sleep(self.request_delay)

        if not chunks:
            logger.warning(
                "No data returned for %s from %s to %s",
                city_name, start_date, end_date,
            )
            return pd.DataFrame()

        df = pd.concat(chunks, ignore_index=True)

        # Deduplicate by timestamp (in case of overlapping chunks)
        if "timestamp" in df.columns:
            df = df.drop_duplicates(subset=["timestamp"], keep="last")
            df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(
            "Completed %s fetch for %s: %d total rows, %d API requests, %d errors",
            self.__class__.__name__, city_name,
            len(df), self._request_count, self._error_count,
        )

        return df

    def fetch_all_cities(
        self,
        city_configs: List[Dict[str, Any]],
        start_date: str,
        end_date: str,
        **kwargs,
    ) -> pd.DataFrame:
        """Download historical data for multiple cities.

        Args:
            city_configs: List of city dicts with 'id', 'name', 'latitude', 'longitude'.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            **kwargs: Provider-specific parameters.

        Returns:
            DataFrame with hourly observations for all cities.
        """
        all_dfs = []

        for city in city_configs:
            df = self.fetch_historical(
                latitude=city["latitude"],
                longitude=city["longitude"],
                location_id=city["id"],
                city_name=city["name"],
                start_date=start_date,
                end_date=end_date,
                **kwargs,
            )
            if not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            return pd.DataFrame()

        return pd.concat(all_dfs, ignore_index=True)

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get API usage summary for this provider session.

        Returns:
            Dictionary with request count, error count.
        """
        return {
            "provider": self.__class__.__name__,
            "total_requests": self._request_count,
            "total_errors": self._error_count,
        }
