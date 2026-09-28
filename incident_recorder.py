#!/usr/bin/env python3
"""
CLI Tool: Rolling Circular Video Buffer & Clip Packager (`incident_recorder.py`)
Usage:
    python incident_recorder.py
"""

from uuid import uuid4
import numpy as np

from src.alerts.incident_recorder import IncidentClipRecorder
from src.utils.logger import setup_logger

logger = setup_logger("IncidentRecorderCLI")


def main():
    recorder = IncidentClipRecorder(storage_dir="data/incidents", pre_incident_sec=2.0, post_incident_sec=2.0, fps=10.0)

    logger.info("Simulating pre-incident ring-buffer frame captures...")
    cam_id = "CAM_TEST_01"

    for i in range(20):
        frame = np.full((720, 1280, 3), 40 + i * 5, dtype=np.uint8)
        recorder.push_frame(cam_id, frame)

    post_frames = [np.full((720, 1280, 3), 150, dtype=np.uint8) for _ in range(20)]

    inc_id = str(uuid4())
    bbox = [300.0, 200.0, 450.0, 500.0]

    clip_p, key_p = recorder.record_incident_clip(
        camera_id=cam_id,
        incident_id=inc_id,
        post_frames=post_frames,
        gesture_name="Fall",
        confidence=0.96,
        bbox=bbox
    )

    logger.info(f"Clip Recording Test PASSED: Clip -> {clip_p} | Keyframe -> {key_p}")


if __name__ == "__main__":
    main()
