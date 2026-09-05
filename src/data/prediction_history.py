"""
Prediction History Store

SQLite-based storage for prediction history.
Provides query, store, and cleanup for past predictions.
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PredictionHistoryStore:
    """
    SQLite-backed prediction history.

    Stores every prediction made by the API for audit, analysis,
    and future performance monitoring.
    """

    def __init__(self, db_path: str = "data/predictions/prediction_history.db"):
        """
        Initialize prediction history store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    city TEXT NOT NULL,
                    aqi_24h INTEGER,
                    aqi_48h INTEGER,
                    aqi_72h INTEGER,
                    category_24h TEXT,
                    category_48h TEXT,
                    category_72h TEXT,
                    model_version TEXT,
                    data_source TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_predictions_city
                ON predictions(city)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
                ON predictions(timestamp)
            """
            )
            conn.commit()

    def store_prediction(self, prediction: Dict[str, Any]) -> int:
        """
        Store a prediction in the history.

        Args:
            prediction: Prediction response dictionary

        Returns:
            Row ID of stored prediction
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                """INSERT INTO predictions
                   (timestamp, city, aqi_24h, aqi_48h, aqi_72h,
                    category_24h, category_48h, category_72h,
                    model_version, data_source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    prediction.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    prediction.get("city", "unknown"),
                    prediction.get("aqi_24h"),
                    prediction.get("aqi_48h"),
                    prediction.get("aqi_72h"),
                    prediction.get("category_24h"),
                    prediction.get("category_48h"),
                    prediction.get("category_72h"),
                    prediction.get("model_version"),
                    prediction.get("data_source", "open-meteo"),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_history(
        self,
        city: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get prediction history.

        Args:
            city: Filter by city name
            start_date: Filter from date (YYYY-MM-DD)
            end_date: Filter to date (YYYY-MM-DD)
            limit: Maximum number of results

        Returns:
            List of prediction dictionaries
        """
        query = "SELECT * FROM predictions WHERE 1=1"
        params = []

        if city:
            query += " AND LOWER(city) = LOWER(?)"
            params.append(city)

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date + "T23:59:59")

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        return [dict(row) for row in rows]

    def get_stats(self, city: Optional[str] = None) -> Dict[str, Any]:
        """
        Get prediction statistics.

        Args:
            city: Filter by city

        Returns:
            Statistics dictionary
        """
        where = ""
        params = []
        if city:
            where = "WHERE LOWER(city) = LOWER(?)"
            params.append(city)

        with sqlite3.connect(str(self.db_path)) as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM predictions {where}", params).fetchone()[0]

            if total == 0:
                return {"total": 0, "cities": [], "date_range": None}

            cities = [
                row[0]
                for row in conn.execute(
                    f"SELECT DISTINCT city FROM predictions {where}", params
                ).fetchall()
            ]

            date_range = conn.execute(
                f"SELECT MIN(timestamp), MAX(timestamp) FROM predictions {where}",
                params,
            ).fetchone()

            avg_aqi = conn.execute(
                (
                    f"SELECT AVG(aqi_24h) FROM predictions {where} AND aqi_24h IS NOT NULL"
                    if city
                    else "SELECT AVG(aqi_24h) FROM predictions WHERE aqi_24h IS NOT NULL"
                ),
                params,
            ).fetchone()[0]

        return {
            "total": total,
            "cities": cities,
            "date_range": {
                "start": date_range[0],
                "end": date_range[1],
            },
            "average_aqi_24h": round(avg_aqi, 1) if avg_aqi else None,
        }

    def cleanup_old(self, days: int = 90) -> int:
        """
        Remove predictions older than N days.

        Args:
            days: Number of days to keep

        Returns:
            Number of rows deleted
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("DELETE FROM predictions WHERE timestamp < ?", (cutoff,))
            conn.commit()
            return cursor.rowcount
