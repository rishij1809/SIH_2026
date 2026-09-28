#!/usr/bin/env python3
"""
CLI Tool: Asynchronous Task Queue & Dispatch Workers (`dispatch_worker.py`)
Usage:
    python dispatch_worker.py
"""

from src.alerts.dispatch_worker import AlertDispatcher
from src.alerts.schemas import IncidentPayload, IncidentType, SpatialCoordinates
from src.utils.logger import setup_logger

logger = setup_logger("DispatchWorkerCLI")


def main():
    dispatcher = AlertDispatcher(c2_webhook_url="https://httpbin.org/post", enable_twilio=True, enable_local_alarm=True)

    payload = IncidentPayload(
        camera_id="CAM_NORTH_02",
        location_tag="North Sector Gate",
        event_type=IncidentType.FALL,
        confidence_score=0.96,
        track_id=45,
        spatial_coordinates=SpatialCoordinates(x=200.0, y=300.0, w=100.0, h=250.0),
        clip_path="data/incidents/clips/demo_fall.mp4",
        keyframe_path="data/incidents/keyframes/demo_fall.jpg"
    )

    logger.info("Executing test alert dispatch...")
    dispatcher.dispatch_incident_sync(payload)

    logger.info("Alert Dispatcher test PASSED successfully!")


if __name__ == "__main__":
    main()
