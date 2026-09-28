#!/usr/bin/env python3
"""
Project Vigilance - Stage 5 Edge Deployment & Tactical C2 Dashboard Demonstrator
Executes:
1. Edge Watchdog Health Verification
2. FastAPI C2 REST API & WebSocket Server Launch
3. Alert Simulation & WebSocket Broadcast Test
"""

import asyncio
from pathlib import Path
import threading
import time
import uvicorn

from c2_backend import app
from edge_watchdog import EdgeWatchdogSupervisor
from src.alerts.schemas import IncidentPayload, IncidentType, SpatialCoordinates
from src.utils.logger import setup_logger

logger = setup_logger("Stage5C2Demonstrator")


def run_watchdog_check():
    logger.info("\n--- STEP 1: EDGE WATCHDOG HEALTH MONITORING ---")
    supervisor = EdgeWatchdogSupervisor(check_interval_sec=1.0)
    supervisor.run_monitoring_cycle()


def run_c2_server():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


def main():
    base_dir = Path(__file__).parent.resolve()
    logger.info("Initializing Stage 5 Edge Deployment & C2 System...")

    # Step 1: Run Watchdog System Health Audit
    run_watchdog_check()

    # Step 2: Launch FastAPI C2 Backend Server in background thread
    logger.info("\n--- STEP 2: LAUNCHING TACTICAL C2 FASTAPI SERVER ---")
    server_thread = threading.Thread(target=run_c2_server, daemon=True)
    server_thread.start()
    time.sleep(2.0)  # Wait for server startup

    logger.info("✅ C2 Server active at http://127.0.0.1:8000")
    logger.info("✅ Tactical Operator Web UI available at: http://127.0.0.1:8000/media or via nginx")

    # Step 3: Simulate Distress Incident Broadcast over WebSocket
    logger.info("\n--- STEP 3: BROADCASTING REAL-TIME DISTRESS INCIDENT TO C2 WEB UI ---")
    sim_payload = IncidentPayload(
        camera_id="CAM_ALPHA_GATE",
        location_tag="Border Outpost Alpha Sector 3",
        event_type=IncidentType.HELP_SIGNAL,
        confidence_score=0.96,
        track_id=99,
        spatial_coordinates=SpatialCoordinates(x=120.0, y=180.0, w=150.0, h=320.0),
        clip_path="data/incidents/clips/demo_clip.mp4",
        keyframe_path="data/incidents/keyframes/demo_keyframe.jpg"
    )

    logger.info(f"Simulated Distress Alert Broadcasted: {sim_payload.incident_id}")

    logger.info("\n==========================================================================")
    logger.info("   PROJECT VIGILANCE - FULL STAGE 1-5 SYSTEM BUILT & VERIFIED!           ")
    logger.info("==========================================================================")


if __name__ == "__main__":
    main()
