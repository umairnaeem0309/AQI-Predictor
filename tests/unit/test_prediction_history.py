"""
Tests for prediction history store.
"""

import os
import sqlite3
import tempfile
from datetime import datetime, timezone

import pytest

from src.data.prediction_history import PredictionHistoryStore


class TestPredictionHistoryStore:
    """Test SQLite prediction history store."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a temporary store for testing."""
        db_path = str(tmp_path / "test_predictions.db")
        return PredictionHistoryStore(db_path=db_path)

    @pytest.fixture
    def sample_prediction(self):
        return {
            "city": "Karachi",
            "timestamp": "2026-08-28T12:00:00+00:00",
            "aqi_24h": 137,
            "aqi_48h": 79,
            "aqi_72h": 138,
            "category_24h": "Unhealthy for Sensitive Groups",
            "category_48h": "Moderate",
            "category_72h": "Unhealthy for Sensitive Groups",
            "model_version": "xgboost-v1.0",
            "data_source": "open-meteo",
        }

    def test_init_creates_db(self, store):
        """Test that initialization creates the database file."""
        assert os.path.exists(store.db_path)

    def test_init_creates_tables(self, store):
        """Test that tables are created."""
        with sqlite3.connect(str(store.db_path)) as conn:
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            table_names = [t[0] for t in tables]
            assert "predictions" in table_names

    def test_store_prediction(self, store, sample_prediction):
        """Test storing a prediction."""
        row_id = store.store_prediction(sample_prediction)
        assert row_id is not None
        assert row_id > 0

    def test_get_history(self, store, sample_prediction):
        """Test retrieving prediction history."""
        store.store_prediction(sample_prediction)
        history = store.get_history()
        assert len(history) == 1
        assert history[0]["city"] == "Karachi"
        assert history[0]["aqi_24h"] == 137

    def test_get_history_by_city(self, store, sample_prediction):
        """Test filtering by city."""
        store.store_prediction(sample_prediction)

        other = sample_prediction.copy()
        other["city"] = "Lahore"
        other["aqi_24h"] = 183
        store.store_prediction(other)

        karachi = store.get_history(city="karachi")
        assert len(karachi) == 1
        assert karachi[0]["city"] == "Karachi"

        lahore = store.get_history(city="Lahore")
        assert len(lahore) == 1
        assert lahore[0]["city"] == "Lahore"

    def test_get_history_limit(self, store, sample_prediction):
        """Test result limiting."""
        for i in range(10):
            pred = sample_prediction.copy()
            pred["aqi_24h"] = 100 + i
            store.store_prediction(pred)

        history = store.get_history(limit=5)
        assert len(history) == 5

    def test_get_history_date_filter(self, store):
        """Test date range filtering."""
        pred1 = {
            "city": "Karachi",
            "timestamp": "2026-08-01T12:00:00+00:00",
            "aqi_24h": 100,
        }
        pred2 = {
            "city": "Karachi",
            "timestamp": "2026-08-28T12:00:00+00:00",
            "aqi_24h": 150,
        }
        store.store_prediction(pred1)
        store.store_prediction(pred2)

        # Get only August 28
        history = store.get_history(start_date="2026-08-28", end_date="2026-08-28")
        assert len(history) == 1
        assert history[0]["aqi_24h"] == 150

    def test_get_stats(self, store, sample_prediction):
        """Test getting statistics."""
        store.store_prediction(sample_prediction)
        stats = store.get_stats()
        assert stats["total"] == 1
        assert "Karachi" in stats["cities"]

    def test_get_stats_empty(self, store):
        """Test stats on empty store."""
        stats = store.get_stats()
        assert stats["total"] == 0

    def test_cleanup_old(self, store):
        """Test old prediction cleanup."""
        # Store an old prediction
        old = {
            "city": "Karachi",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "aqi_24h": 100,
        }
        store.store_prediction(old)

        # Store a recent prediction
        recent = {
            "city": "Karachi",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "aqi_24h": 150,
        }
        store.store_prediction(recent)

        # Cleanup predictions older than 30 days
        deleted = store.cleanup_old(days=30)
        assert deleted == 1

        # Only recent should remain
        history = store.get_history()
        assert len(history) == 1
        assert history[0]["aqi_24h"] == 150

    def test_multiple_cities(self, store, sample_prediction):
        """Test storing predictions for multiple cities."""
        for city in ["Karachi", "Lahore", "Islamabad"]:
            pred = sample_prediction.copy()
            pred["city"] = city
            store.store_prediction(pred)

        stats = store.get_stats()
        assert stats["total"] == 3
        assert len(stats["cities"]) == 3

    def test_history_ordering(self, store):
        """Test that history returns most recent first."""
        for i in range(5):
            pred = {
                "city": "Karachi",
                "timestamp": f"2026-08-{25 + i}T12:00:00+00:00",
                "aqi_24h": 100 + i,
            }
            store.store_prediction(pred)

        history = store.get_history()
        timestamps = [h["timestamp"] for h in history]
        assert timestamps == sorted(timestamps, reverse=True)


class TestAPIClientHistory:
    """Test API client history methods."""

    def test_history_mock(self):
        from app.frontend.utils.api_client import APIClient

        client = APIClient(mock_mode=True)
        # No history mock yet — just verify the client works
        assert client.mock_mode is True
