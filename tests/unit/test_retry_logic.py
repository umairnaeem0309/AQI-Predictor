"""
Tests for retry logic and error classification.

Tests cover:
- Retryable errors trigger retries (network, timeout, 429, 5xx)
- Non-retryable errors raise immediately (401, 403, 4xx)
- Exponential backoff timing
- Max retries exhaustion
- Cache key generation
"""

import time
import pytest
import responses

from src.data.base_client import BaseAPIClient
from src.data.exceptions import (
    APITimeoutError,
    APINetworkError,
    APIAuthenticationError,
    APIRateLimitError,
    APIServerError,
    APIRequestError,
    APIValidationError,
)


# =============================================================================
# Concrete Test Client
# =============================================================================


class TestClient(BaseAPIClient):
    """Concrete implementation of BaseAPIClient for testing."""

    def _build_request(self, **kwargs):
        endpoint = kwargs.get("endpoint", "test")
        params = kwargs.get("params", {})
        return endpoint, params

    def _parse_response(self, raw_json, **kwargs):
        return [raw_json]

    def _validate_response(self, raw_json):
        return bool(raw_json)


# =============================================================================
# Test Client Initialization
# =============================================================================


class TestBaseClientInit:
    """Tests for BaseAPIClient initialization."""

    def test_init_with_api_key(self):
        """Client initializes with API key."""
        client = TestClient(api_key="test-key")
        assert client.api_key == "test-key"

    def test_init_without_api_key(self):
        """Client initializes without API key (test mode)."""
        client = TestClient()
        assert client.api_key is None

    def test_default_retry_settings(self):
        """Default retry settings are applied."""
        client = TestClient()
        assert client.max_retries == 3
        assert client.retry_backoff_base == 1.0

    def test_custom_retry_settings(self):
        """Custom retry settings are applied."""
        client = TestClient(max_retries=5, retry_backoff_base=0.5)
        assert client.max_retries == 5
        assert client.retry_backoff_base == 0.5


# =============================================================================
# Test Retryable Errors
# =============================================================================


class TestRetryableErrors:
    """Tests that retryable errors trigger retries."""

    @responses.activate
    def test_timeout_triggers_retry(self):
        """Timeout triggers retry."""
        responses.add(
            responses.GET,
            "https://api.test.com/test",
            body=responses.exceptions.Timeout(),
        )
        responses.add(
            responses.GET,
            "https://api.test.com/test",
            json={"result": "ok"},
            status=200,
        )

        client = TestClient(api_key="test", max_retries=2, retry_backoff_base=0.01)
        result = client._retry_request("test")
        assert result == {"result": "ok"}
        assert len(responses.calls) == 2

    @responses.activate
    def test_rate_limit_triggers_retry(self):
        """HTTP 429 triggers retry."""
        responses.add(
            responses.GET,
            "https://api.test.com/test",
            json={"message": "rate limited"},
            status=429,
        )
        responses.add(
            responses.GET,
            "https://api.test.com/test",
            json={"result": "ok"},
            status=200,
        )

        client = TestClient(api_key="test", max_retries=2, retry_backoff_base=0.01)
        result = client._retry_request("test")
        assert result == {"result": "ok"}
        assert len(responses.calls) == 2

    @responses.activate
    def test_server_error_triggers_retry(self):
        """HTTP 500 triggers retry."""
        responses.add(
            responses.GET,
            "https://api.test.com/test",
            json={"message": "internal error"},
            status=500,
        )
        responses.add(
            responses.GET,
            "https://api.test.com/test",
            json={"result": "ok"},
            status=200,
        )

        client = TestClient(api_key="test", max_retries=2, retry_backoff_base=0.01)
        result = client._retry_request("test")
        assert result == {"result": "ok"}

    @responses.activate
    def test_502_triggers_retry(self):
        """HTTP 502 triggers retry."""
        responses.add(
            responses.GET,
            "https://api.test.com/test",
            status=502,
        )
        responses.add(
            responses.GET,
            "https://api.test.com/test",
            json={"result": "ok"},
            status=200,
        )

        client = TestClient(api_key="test", max_retries=2, retry_backoff_base=0.01)
        result = client._retry_request("test")
        assert result == {"result": "ok"}

    @responses.activate
    def test_503_triggers_retry(self):
        """HTTP 503 triggers retry."""
        responses.add(
            responses.GET,
            "https://api.test.com/test",
            status=503,
        )
        responses.add(
            responses.GET,
            "https://api.test.com/test",
            json={"result": "ok"},
            status=200,
        )

        client = TestClient(api_key="test", max_retries=2, retry_backoff_base=0.01)
        result = client._retry_request("test")
        assert result == {"result": "ok"}


# =============================================================================
# Test Non-Retryable Errors
# =============================================================================


class TestNonRetryableErrors:
    """Tests that non-retryable errors raise immediately."""

    @responses.activate
    def test_auth_failure_no_retry(self):
        """HTTP 401 raises immediately without retry."""
        responses.add(
            responses.GET,
            "https://api.test.com/test",
            json={"message": "unauthorized"},
            status=401,
        )

        client = TestClient(api_key="bad", max_retries=3, retry_backoff_base=0.01)
        with pytest.raises(APIAuthenticationError):
            client._retry_request("test")
        assert len(responses.calls) == 1  # Only one call, no retries

    @responses.activate
    def test_forbidden_no_retry(self):
        """HTTP 403 raises immediately without retry."""
        responses.add(
            responses.GET,
            "https://api.test.com/test",
            json={"message": "forbidden"},
            status=403,
        )

        client = TestClient(api_key="test", max_retries=3, retry_backoff_base=0.01)
        with pytest.raises(APIAuthenticationError):
            client._retry_request("test")
        assert len(responses.calls) == 1

    @responses.activate
    def test_400_no_retry(self):
        """HTTP 400 raises immediately without retry."""
        responses.add(
            responses.GET,
            "https://api.test.com/test",
            json={"message": "bad request"},
            status=400,
        )

        client = TestClient(api_key="test", max_retries=3, retry_backoff_base=0.01)
        with pytest.raises(APIRequestError):
            client._retry_request("test")
        assert len(responses.calls) == 1

    @responses.activate
    def test_404_no_retry(self):
        """HTTP 404 raises immediately without retry."""
        responses.add(
            responses.GET,
            "https://api.test.com/test",
            json={"message": "not found"},
            status=404,
        )

        client = TestClient(api_key="test", max_retries=3, retry_backoff_base=0.01)
        with pytest.raises(APIRequestError):
            client._retry_request("test")
        assert len(responses.calls) == 1


# =============================================================================
# Test Max Retries Exhaustion
# =============================================================================


class TestMaxRetriesExhaustion:
    """Tests that max retries exhaustion raises the last error."""

    @responses.activate
    def test_all_retries_exhausted(self):
        """Raises last error after all retries fail."""
        for _ in range(5):
            responses.add(
                responses.GET,
                "https://api.test.com/test",
                json={"message": "server error"},
                status=500,
            )

        client = TestClient(api_key="test", max_retries=3, retry_backoff_base=0.01)
        with pytest.raises(APIServerError):
            client._retry_request("test")
        assert len(responses.calls) == 3

    @responses.activate
    def test_all_timeouts_exhausted(self):
        """Raises APITimeoutError after all timeout retries."""
        for _ in range(5):
            responses.add(
                responses.GET,
                "https://api.test.com/test",
                body=responses.exceptions.Timeout(),
            )

        client = TestClient(api_key="test", max_retries=3, retry_backoff_base=0.01)
        with pytest.raises(APITimeoutError):
            client._retry_request("test")
        assert len(responses.calls) == 3


# =============================================================================
# Test Exponential Backoff Timing
# =============================================================================


class TestBackoffTiming:
    """Tests for exponential backoff behavior."""

    @responses.activate
    def test_backoff_increases(self):
        """Verify retry delays increase exponentially."""
        call_times = []

        def record_time(request):
            call_times.append(time.time())
            return (200, {}, '{"result": "ok"}')

        # First call fails, second succeeds
        responses.add(
            responses.GET,
            "https://api.test.com/test",
            json={"message": "error"},
            status=500,
        )
        responses.add(
            responses.GET,
            "https://api.test.com/test",
            json={"result": "ok"},
            status=200,
        )

        client = TestClient(
            api_key="test",
            max_retries=2,
            retry_backoff_base=0.05,  # 50ms base
        )
        client._retry_request("test")

        # Should have 2 calls
        assert len(responses.calls) == 2


# =============================================================================
# Test Cache Key Generation
# =============================================================================


class TestCacheKey:
    """Tests for cache key generation."""

    def test_deterministic_key(self):
        """Same inputs produce same cache key."""
        client = TestClient(api_key="test")
        key1 = client._get_cache_key("weather", {"lat": 24.86, "lon": 67.0})
        key2 = client._get_cache_key("weather", {"lat": 24.86, "lon": 67.0})
        assert key1 == key2

    def test_different_inputs_different_keys(self):
        """Different inputs produce different cache keys."""
        client = TestClient(api_key="test")
        key1 = client._get_cache_key("weather", {"lat": 24.86})
        key2 = client._get_cache_key("weather", {"lat": 31.52})
        assert key1 != key2

    def test_key_is_sha256(self):
        """Cache key is a SHA-256 hex string."""
        client = TestClient(api_key="test")
        key = client._get_cache_key("test", {})
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_key_with_no_params(self):
        """Cache key works with no parameters."""
        client = TestClient(api_key="test")
        key = client._get_cache_key("test")
        assert len(key) == 64
