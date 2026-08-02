"""
Custom exceptions for data collection layer.

Error hierarchy:
    AQIPredictorError (base)
    ├── APIClientError
    │   ├── APIRateLimitError        — HTTP 429 (retryable)
    │   ├── APIAuthenticationError   — HTTP 401/403 (not retryable)
    │   ├── APIRequestError          — HTTP 4xx (not retryable)
    │   └── APIServerError           — HTTP 5xx (retryable)
    ├── APITimeoutError              — Request timeout (retryable)
    ├── APINetworkError              — Connection failure (retryable)
    ├── APIValidationError           — Response missing fields (not retryable)
    └── DataQualityWarning
        ├── StalenessWarning         — Data older than threshold
        └── DuplicateWarning         — Duplicate records detected
"""


class AQIPredictorError(Exception):
    """Base exception for AQI Predictor project."""

    pass


class APIClientError(AQIPredictorError):
    """General API client error."""

    def __init__(self, message: str, status_code: int = None, response_body: str = None):
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(message)


class APIRateLimitError(APIClientError):
    """HTTP 429 — Rate limit exceeded. Retryable."""

    pass


class APIAuthenticationError(APIClientError):
    """HTTP 401/403 — Authentication failed. Not retryable."""

    pass


class APIRequestError(APIClientError):
    """HTTP 4xx — Invalid request. Not retryable."""

    pass


class APIServerError(APIClientError):
    """HTTP 5xx — Server error. Retryable."""

    pass


class APITimeoutError(APIClientError):
    """Request timed out. Retryable."""

    pass


class APINetworkError(APIClientError):
    """Network/connection failure. Retryable."""

    pass


class APIValidationError(APIClientError):
    """Response missing required fields. Not retryable."""

    pass


class DataQualityWarning(AQIPredictorError):
    """Base warning for data quality issues."""

    pass


class StalenessWarning(DataQualityWarning):
    """Data is older than acceptable threshold."""

    def __init__(self, message: str, data_age_hours: float, max_age_hours: float):
        self.data_age_hours = data_age_hours
        self.max_age_hours = max_age_hours
        super().__init__(message)


class DuplicateWarning(DataQualityWarning):
    """Duplicate records detected in dataset."""

    def __init__(self, message: str, duplicate_count: int):
        self.duplicate_count = duplicate_count
        super().__init__(message)
