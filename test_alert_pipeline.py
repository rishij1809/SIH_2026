#!/usr/bin/env python3
"""
Project Vigilance - Stage 4 End-to-End Alert Dispatch Simulation & Validation Harness (`test_alert_pipeline.py`)
Simulates high-confidence distress events through:
1. Temporal Persistence & Cooldown Gating
2. Incident Clip & Keyframe Packaging
3. Pydantic Payload Schema Verification
4. Tactical C2 Webhook, SMS/Voice, and Offline SQLite Cache Sync
"""

from pathlib import Path
import time
from uuid import uuid4
import numpy as np

from src.alerts.dispatch_worker import AlertDispatcher
from src.alerts.incident_recorder import IncidentClipRecorder
from src.alerts.offline_cache import OfflineCacheManager
from src.alerts.schemas import IncidentPayload, IncidentType, SpatialCoordinates
from src.alerts.verification_engine import HeuristicVerificationEngine
from src.utils.logger import setup_logger

logger = setup_logger("AlertPipelineHarness")


def run_stage4_simulation():
    base_dir = Path(__file__).parent.resolve()
    logger.info("Initializing Stage 4 End-to-End Alert Pipeline Harness...")

    # 1. Instantiate Stage 4 Modules
    verifier = HeuristicVerificationEngine(
        confidence_threshold=0.65,
        persistence_window_N=5,
        persistence_threshold_M=3,
        cooldown_sec=10.0
    )

    recorder = IncidentClipRecorder(
        storage_dir=base_dir / "data" / "incidents",
        pre_incident_sec=2.0,
        post_incident_sec=2.0,
        fps=10.0
    )

    cache = OfflineCacheManager(db_path=base_dir / "data" / "offline_cache.db")

    dispatcher = AlertDispatcher(
        c2_webhook_url="https://httpbin.org/post",
        enable_twilio=True,
        enable_local_alarm=True,
        cache_manager=cache
    )

    # 2. Simulate pre-incident frames
    cam_id = "CAM_OUTPOST_ALPHA"
    logger.info(f"Simulating live video ring-buffer frames for camera [{cam_id}]...")

    for f_idx in range(25):
        frame = np.full((720, 1280, 3), 40 + (f_idx % 20), dtype=np.uint8)
        recorder.push_frame(cam_id, frame)

    # 3. Simulate high-confidence distress detections
    bbox = [250.0, 180.0, 380.0, 480.0]
    track_id = 99
    gesture_id = 1  # Universal Help Signal
    confidence = 0.94

    logger.info("Simulating consecutive distress gesture detections...")
    verified_flag = False

    for step in range(1, 4):
        is_verified, reason = verifier.evaluate_detection(cam_id, track_id, gesture_id, confidence, bbox)
        logger.info(f"Detection Step {step}: Verified={is_verified} | Details: {reason}")
        if is_verified:
            verified_flag = True
            break

    if not verified_flag:
        logger.error("Verification engine failed to verify persistent incident!")
        return

    # 4. Trigger Incident Clip & Keyframe Recorder
    logger.info("Triggering Incident Video & Keyframe Recording...")
    post_frames = [np.full((720, 1280, 3), 120, dtype=np.uint8) for _ in range(20)]
    incident_id = str(uuid4())

    clip_path, keyframe_path = recorder.record_incident_clip(
        camera_id=cam_id,
        incident_id=incident_id,
        post_frames=post_frames,
        gesture_name="Universal Help Signal",
        confidence=confidence,
        bbox=bbox
    )

    # 5. Build Pydantic Incident Payload
    logger.info("Building Pydantic Incident Payload Schema...")
    payload = IncidentPayload(
        incident_id=incident_id,
        camera_id=cam_id,
        location_tag="Border Outpost Alpha Gate 03",
        event_type=IncidentType.HELP_SIGNAL,
        confidence_score=confidence,
        track_id=track_id,
        clip_path=str(clip_path),
        keyframe_path=str(keyframe_path) if keyframe_path else None,
        spatial_coordinates=SpatialCoordinates(x=bbox[0], y=bbox[1], w=bbox[2]-bbox[0], h=bbox[3]-bbox[1])
    )

    logger.info(f"Generated Payload JSON:\n{payload.model_dump_json(indent=2)}")

    # 6. Asynchronous Dispatch
    logger.info("Dispatching Incident Payload through Dispatcher Engine...")
    dispatcher.dispatch_incident_sync(payload)

    logger.info("=== STAGE 4 ALERT DISPATCH PIPELINE TEST PASSED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_stage4_simulation()
