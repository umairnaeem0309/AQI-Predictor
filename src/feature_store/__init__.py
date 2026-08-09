"""
Feature Store — Abstraction layer for feature storage.

Provides:
- FeatureStoreInterface: Abstract base class
- HopsworksStore: Cloud feature store (primary)
- LocalStore: DuckDB + Parquet fallback
- Factory function: get_feature_store()
"""

import os
import logging

from src.feature_store.base import FeatureStoreInterface
from src.feature_store.local_store import LocalStore
from src.feature_store.schemas import (
    DatasetMetadata,
    DatasetType,
    FeatureSchema,
    FeatureGroupMetadata,
    LineageMetadata,
    get_feature_group_name,
)

logger = logging.getLogger(__name__)


def get_feature_store(
    prefer_local: bool = False,
) -> FeatureStoreInterface:
    """Factory function to get the appropriate feature store.

    Logic:
    1. If prefer_local=True, always return LocalStore
    2. If HOPSWORKS_HOST is configured, try HopsworksStore
    3. If Hopsworks connection fails, fall back to LocalStore
    4. If HOPSWORKS_HOST is not configured, use LocalStore

    Args:
        prefer_local: If True, force local storage.

    Returns:
        FeatureStoreInterface implementation.
    """
    if prefer_local:
        logger.info("Using local feature store (preferred)")
        store = LocalStore()
        store.connect()
        return store

    # Check if Hopsworks is configured
    hopsworks_host = os.environ.get("HOPSWORKS_HOST")
    if not hopsworks_host:
        logger.info("HOPSWORKS_HOST not configured — using local feature store")
        store = LocalStore()
        store.connect()
        return store

    # Try Hopsworks
    try:
        from src.feature_store.hopsworks_store import HopsworksStore, ConfigurationError
        store = HopsworksStore()
        store.connect()
        logger.info("Connected to Hopsworks feature store")
        return store
    except ConfigurationError as e:
        logger.warning(
            "Hopsworks configuration error: %s. Falling back to local store.",
            str(e),
        )
    except Exception as e:
        logger.warning(
            "Hopsworks connection failed: %s. Falling back to local store.",
            str(e),
        )

    # Fallback to local
    store = LocalStore()
    store.connect()
    return store


__all__ = [
    "FeatureStoreInterface",
    "HopsworksStore",
    "LocalStore",
    "DatasetMetadata",
    "DatasetType",
    "FeatureSchema",
    "FeatureGroupMetadata",
    "LineageMetadata",
    "get_feature_group_name",
    "get_feature_store",
]
