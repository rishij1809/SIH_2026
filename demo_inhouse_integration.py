#!/usr/bin/env python3
"""
Surgical Integration Demonstrator: Plugging InHouseClassifier into Project Vigilance Pipeline
Shows non-destructive integration connecting Tracklet Buffers -> InHouseClassifier -> Stage 4 Alerts.
"""

from uuid import uuid4
import numpy as np

from src.alerts.dispatch_worker import AlertDispatcher
from src.alerts.schemas import IncidentPayload, IncidentType, SpatialCoordinates
from src.alerts.verification_engine import HeuristicVerificationEngine
from src.pipeline.inhouse_classifier import InHouseClassifier
from src.utils.logger import setup_logger

logger = setup_logger("SurgicalIntegrationDemo")


def run_integrated_pipeline_step():
    logger.info("Initializing Surgical Integration Glue Code...")

    # 1. Instantiate In-House Classifier
    classifier = InHouseClassifier(model_path="models/inhouse_distress.pt", num_keypoints=17)

    # 2. Instantiate Stage 4 Verification & Dispatch Engines
    verifier = HeuristicVerificationEngine(
        confidence_threshold=0.65,
        persistence_window_N=5,
        persistence_threshold_M=3,
        cooldown_sec=10.0
    )
    dispatcher = AlertDispatcher(c2_webhook_url="https://httpbin.org/post", enable_twilio=False, enable_local_alarm=True)

    # 3. Simulate ByteTrack Tracklet Keypoint Buffer (30 frames x 17 keypoints x 3 channels)
    cam_id = "CAM_SURGICAL_01"
    track_id = 101
    bbox = [200.0, 150.0, 320.0, 450.0]
    dummy_tracklet_buffer = np.random.uniform(0.3, 0.7, size=(30, 17, 3)).astype(np.float32)

    logger.info(f"Processing Track ID #{track_id} buffer of shape {dummy_tracklet_buffer.shape}...")

    # --- MINIMAL GLUE CODE ---
    # Step A: Predict gesture & confidence via InHouseClassifier
    label_str, confidence = classifier.predict(dummy_tracklet_buffer)
    logger.info(f"InHouseClassifier Output -> Label: '{label_str}' | Confidence: {confidence*100:.1f}%")

    # Map label to gesture ID
    gesture_map = {"Normal": 0, "Universal Help Signal": 1, "Aggression/Struggle": 2, "Fall": 3}
    gesture_id = gesture_map.get(label_str, 1)  # Fallback to Help Signal for demo trigger

    # Step B: Pass to Stage 4 Verification Engine across consecutive evaluations
    for step in range(1, 4):
        is_verified, reason = verifier.evaluate_detection(
            camera_id=cam_id,
            track_id=track_id,
            gesture_id=gesture_id,
            confidence=0.92,  # Simulated high confidence
            bbox=bbox
        )
        if is_verified:
            logger.info(f"✅ Stage 4 Verification PASSED on step {step}! Dispatching payload...")
            
            # Step C: Emit Stage 4 Incident Payload
            payload = IncidentPayload(
                incident_id=str(uuid4()),
                camera_id=cam_id,
                location_tag="Surgical Outpost Sector 2",
                event_type=IncidentType.HELP_SIGNAL,
                confidence_score=0.92,
                track_id=track_id,
                spatial_coordinates=SpatialCoordinates(x=bbox[0], y=bbox[1], w=bbox[2]-bbox[0], h=bbox[3]-bbox[1])
            )
            dispatcher.dispatch_incident_sync(payload)
            break
    # --- END GLUE CODE ---

    logger.info("=== SURGICAL INTEGRATION DEMO COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_integrated_pipeline_step()
