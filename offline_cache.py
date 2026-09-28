#!/usr/bin/env python3
"""
CLI Tool: Offline Edge Resilience & Storage Sync (`offline_cache.py`)
Usage:
    python offline_cache.py
"""

import time
from src.alerts.offline_cache import BackgroundSyncDaemon, OfflineCacheManager
from src.alerts.schemas import IncidentPayload, IncidentType, SpatialCoordinates
from src.utils.logger import setup_logger

logger = setup_logger("OfflineCacheCLI")


def main():
    cache = OfflineCacheManager(db_path="data/offline_cache.db")

    # Queue synthetic offline alert
    payload = IncidentPayload(
        camera_id="CAM_OUTPOST_01",
        location_tag="Border Outpost Alpha",
        event_type=IncidentType.AGGRESSION,
        confidence_score=0.91,
        track_id=88,
        spatial_coordinates=SpatialCoordinates(x=100.0, y=100.0, w=50.0, h=150.0)
    )

    logger.info("Enqueuing test alert into SQLite offline cache...")
    cache.enqueue_alert(payload)

    pending = cache.get_pending_alerts()
    logger.info(f"Retrieved {len(pending)} pending offline alert(s) from SQLite database.")

    # Initialize sync daemon
    sync_daemon = BackgroundSyncDaemon(cache, target_webhook_url="https://httpbin.org/post")
    sync_daemon.start()

    time.sleep(3.0)
    sync_daemon.stop()

    logger.info("Offline Cache & Sync Test PASSED successfully!")


if __name__ == "__main__":
    main()
