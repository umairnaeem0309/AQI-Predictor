"""
Base API Client — Abstract foundation for all data source clients.

Features:
- Exponential backoff retry for retryable errors (network, timeout, 429, 5xx)
- No retry for non-retryable errors (401, 403, 4xx)
- Dependency injection for API keys (supports initialization without credentials)
- Caching readiness: cache_key() method for future request deduplication
- Comprehensive logging of all API interactions
"""

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import requests

from src.data.exceptions import (
    APIClientError,
    APIRateLimitError,
    APIServerError,
    APITimeoutError,
    APINetworkError,
    APIAuthenticationError,
    APIRequestError,
)

logger = logging.getLogger(__name__)

# HTTP status codes that should trigger retry
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class BaseAPIClient(ABC):
    """Abstract base class for API clients.

    Subclasses must implement:
    - _parse_response(): Transform raw JSON to StandardObservation
    - _validate_response(): Check required fields are present

    Attributes:
        api_key: API authentication key (None for test/mock mode).
        base_url: Base URL for the API.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retry attempts.
        retry_backoff_base: Base delay for exponential backoff.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "",
        timeout: int = 30,
        max_retries: int = 3,
        retry_backoff_base: float = 1.0,
    ):
        """Initialize the API client.

        Args:
            api_key: API key for authentication. None for test/mock mode.
            base_url: Base URL for the API.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts for retryable errors.
            retry_backoff_base: Base delay in seconds for exponential backoff.
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base
        self._session = requests.Session()

        if not self.api_key:
            logger.warning(
                "%s initialized without API key — requests will fail authentication",
                self.__class__.__name__,
            )

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make an HTTP GET request with timeout handling.

        Args:
            endpoint: API endpoint path (appended to base_url).
            params: Query parameters.

        Returns:
            Parsed JSON response as dictionary.

        Raises:
            APITimeoutError: Request timed out.
            APINetworkError: Connection failure.
            APIAuthenticationError: 401/403 response.
            APIRequestError: Other 4xx response.
            APIServerError: 5xx response.
            APIClientError: Other HTTP errors.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.debug("Making request to %s with params %s", url, params)

        try:
            response = self._session.get(url, params=params, timeout=self.timeout)
        except requests.exceptions.Timeout:
            raise APITimeoutError(f"Request to {url} timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError as e:
            raise APINetworkError(f"Connection failed to {url}: {e}")
        except requests.exceptions.RequestException as e:
            raise APIClientError(f"Request failed: {e}")

        # Log response status
        logger.debug(
            "Response from %s: status=%d, content_length=%d",
            url,
            response.status_code,
            len(response.content),
        )

        # Handle non-retryable errors first (401, 403, 4xx)
        if response.status_code in (401, 403):
            raise APIAuthenticationError(
                f"Authentication failed for {url} (HTTP {response.status_code})",
                status_code=response.status_code,
                response_body=response.text,
            )

        if 400 <= response.status_code < 500 and response.status_code not in RETRYABLE_STATUS_CODES:
            raise APIRequestError(
                f"Invalid request to {url} (HTTP {response.status_code})",
                status_code=response.status_code,
                response_body=response.text,
            )

        # Handle rate limiting (retryable)
        if response.status_code == 429:
            raise APIRateLimitError(
                f"Rate limited by {url} (HTTP 429)",
                status_code=429,
                response_body=response.text,
            )

        # Handle server errors (retryable)
        if response.status_code >= 500:
            raise APIServerError(
                f"Server error from {url} (HTTP {response.status_code})",
                status_code=response.status_code,
                response_body=response.text,
            )

        # Handle other non-2xx responses
        if response.status_code != 200:
            raise APIClientError(
                f"Unexpected response from {url} (HTTP {response.status_code})",
                status_code=response.status_code,
                response_body=response.text,
            )

        try:
            return response.json()
        except json.JSONDecodeError as e:
            raise APIClientError(f"Invalid JSON response from {url}: {e}")

    def _retry_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make a request with exponential backoff retry.

        Retries only on retryable errors:
        - Network failures (APINetworkError)
        - Timeouts (APITimeoutError)
        - HTTP 429 (APIRateLimitError)
        - HTTP 5xx (APIServerError)

        Does NOT retry:
        - Authentication failures (401/403)
        - Invalid requests (4xx)

        Args:
            endpoint: API endpoint path.
            params: Query parameters.

        Returns:
            Parsed JSON response as dictionary.

        Raises:
            The last exception after all retries exhausted.
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return self._make_request(endpoint, params)
            except (APINetworkError, APITimeoutError, APIRateLimitError, APIServerError) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_backoff_base * (2 ** attempt)
                    logger.warning(
                        "Request attempt %d/%d failed: %s. Retrying in %.1fs...",
                        attempt + 1,
                        self.max_retries,
                        str(e),
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "All %d retry attempts exhausted for %s. Last error: %s",
                        self.max_retries,
                        endpoint,
                        str(e),
                    )
            except (APIAuthenticationError, APIRequestError, APIValidationError):
                # Non-retryable errors — raise immediately
                raise

        raise last_exception

    @abstractmethod
    def _parse_response(
        self, raw_json: Dict[str, Any], **kwargs
    ):
        """Transform raw API JSON into StandardObservation(s).

        Subclasses must implement this to handle their specific
        response format.

        Args:
            raw_json: Parsed JSON response from the API.
            **kwargs: Additional context (city_id, city_name, etc.).

        Returns:
            List of StandardObservation objects.
        """
        pass

    @abstractmethod
    def _validate_response(self, raw_json: Dict[str, Any]) -> bool:
        """Check that the response contains required fields.

        Args:
            raw_json: Parsed JSON response from the API.

        Returns:
            True if response is valid, False otherwise.
        """
        pass

    def _get_cache_key(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Generate a cache key for a request.

        Designed for future caching integration. The key is a deterministic
        hash of the endpoint and sorted parameters.

        Args:
            endpoint: API endpoint path.
            params: Query parameters.

        Returns:
            SHA-256 hash string suitable for use as cache key.
        """
        key_data = json.dumps({"endpoint": endpoint, "params": params or {}}, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()

    def fetch_data(self, **kwargs) -> list:
        """Public entry point for data fetching.

        Flow: request → retry → validate → parse → return

        Args:
            **kwargs: Client-specific parameters (city_id, city_coord, etc.).

        Returns:
            List of StandardObservation objects.

        Raises:
            APIClientError: If request fails after retries.
            APIValidationError: If response validation fails.
        """
        endpoint, params = self._build_request(**kwargs)

        logger.info(
            "Fetching data from %s endpoint=%s params=%s",
            self.__class__.__name__,
            endpoint,
            params,
        )

        raw_json = self._retry_request(endpoint, params)

        if not self._validate_response(raw_json):
            from src.data.exceptions import APIValidationError

            raise APIValidationError(
                f"Response validation failed for {endpoint}"
            )

        observations = self._parse_response(raw_json, **kwargs)

        logger.info(
            "Successfully fetched %d observations from %s",
            len(observations),
            self.__class__.__name__,
        )

        return observations

    @abstractmethod
    def _build_request(self, **kwargs):
        """Build endpoint and parameters for the API request.

        Args:
            **kwargs: Client-specific parameters.

        Returns:
            Tuple of (endpoint, params).
        """
        pass
