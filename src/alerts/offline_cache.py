"""
Offline Edge Resilience & Storage Sync Engine for Project Vigilance.
Serializes alert payloads into SQLite queue database when WAN/SATCOM is degraded,
and runs a background sync daemon that flushes cached alerts once connectivity returns.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Union
import requests

from src.alerts.schemas import IncidentPayload
from src.utils.logger import setup_logger

logger = setup_logger("OfflineCacheManager")


class OfflineCacheManager:
    """SQLite-backed persistent queue for offline edge resilience."""

    def __init__(self, db_path: Union[str, Path] = "data/offline_cache.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initializes SQLite alerts_queue schema table."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    incident_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT DEFAULT 'PENDING',
                    attempts INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.commit()

    def enqueue_alert(self, payload: IncidentPayload) -> bool:
        """Serializes IncidentPayload into SQLite persistent queue table."""
        try:
            payload_json = payload.model_dump_json()
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO alerts_queue (incident_id, timestamp, camera_id, payload_json, status)
                    VALUES (?, ?, ?, ?, 'PENDING')
                    """,
                    (payload.incident_id, payload.timestamp, payload.camera_id, payload_json)
                )
                conn.commit()
            logger.info(f"Queued alert [{payload.incident_id}] in local SQLite offline cache.")
            return True
        except Exception as e:
            logger.error(f"Failed to enqueue alert into SQLite cache: {e}")
            return False

    def get_pending_alerts(self, limit: int = 50) -> List[IncidentPayload]:
        """Retrieves list of unsent PENDING alerts."""
        payloads = []
        try:
            with self._get_connection() as conn:
                rows = conn.execute(
                    "SELECT payload_json FROM alerts_queue WHERE status = 'PENDING' ORDER BY id ASC LIMIT ?",
                    (limit,)
                ).fetchall()

                for row in rows:
                    p = IncidentPayload.model_validate_json(row["payload_json"])
                    payloads.append(p)
        except Exception as e:
            logger.error(f"Failed to fetch pending alerts from SQLite: {e}")
        return payloads

    def mark_as_synced(self, incident_id: str) -> None:
        """Updates alert status to SYNCED."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE alerts_queue SET status = 'SYNCED' WHERE incident_id = ?",
                (incident_id,)
            )
            conn.commit()
        logger.info(f"Marked alert [{incident_id}] as SYNCED in SQLite cache.")

    def increment_attempts(self, incident_id: str) -> None:
        """Increments failed sync attempt counter."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE alerts_queue SET attempts = attempts + 1 WHERE incident_id = ?",
                (incident_id,)
            )
            conn.commit()


class BackgroundSyncDaemon:
    """Monitors WAN connectivity and flushes cached offline alerts to central command server."""

    def __init__(
        self,
        cache_manager: OfflineCacheManager,
        target_webhook_url: str = "https://httpbin.org/post",
        check_interval_sec: float = 10.0
    ) -> None:
        self.cache_manager = cache_manager
        self.target_webhook_url = target_webhook_url
        self.check_interval_sec = check_interval_sec

        self.running = False
        self.thread: Optional[threading.Thread] = None

    def check_wan_connectivity(self) -> bool:
        """Pings target webhook or WAN DNS to test backhaul availability."""
        try:
            resp = requests.head("https://1.1.1.1", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    def flush_pending_alerts(self) -> int:
        """Sends all pending offline alerts to central C2 command server."""
        pending = self.cache_manager.get_pending_alerts()
        if not pending:
            return 0

        synced_count = 0
        logger.info(f"WAN connection restored. Flushing {len(pending)} cached alerts to server...")

        for payload in pending:
            try:
                resp = requests.post(
                    self.target_webhook_url,
                    json=payload.model_dump(),
                    timeout=5.0
                )
                if resp.status_code in [200, 201, 202]:
                    self.cache_manager.mark_as_synced(payload.incident_id)
                    synced_count += 1
                else:
                    self.cache_manager.increment_attempts(payload.incident_id)
            except Exception as e:
                self.cache_manager.increment_attempts(payload.incident_id)

        logger.info(f"Sync complete: {synced_count}/{len(pending)} alerts delivered.")
        return synced_count

    def _sync_loop(self) -> None:
        """Worker loop running periodic connectivity check & sync."""
        while self.running:
            if self.check_wan_connectivity():
                self.flush_pending_alerts()
            time.sleep(self.check_interval_sec)

    def start(self) -> None:
        """Launches background sync daemon thread."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.thread.start()
        logger.info("Background WAN Sync Daemon launched.")

    def stop(self) -> None:
        """Stops background sync daemon thread."""
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        logger.info("Background WAN Sync Daemon stopped.")
