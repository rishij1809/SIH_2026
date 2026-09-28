#!/usr/bin/env python3
"""
CLI / Importable Schema Module for Project Vigilance Stage 4 (`schemas.py`)
"""

from src.alerts.schemas import IncidentPayload, IncidentType, SpatialCoordinates


def test_schemas():
    payload = IncidentPayload(
        camera_id="CAM_ALPHA_01",
        location_tag="Sector 4 Outpost",
        event_type=IncidentType.HELP_SIGNAL,
        confidence_score=0.94,
        track_id=12,
        spatial_coordinates=SpatialCoordinates(x=120.0, y=200.0, w=80.0, h=180.0),
        clip_path="data/incidents/clips/demo.mp4",
        keyframe_path="data/incidents/keyframes/demo.jpg"
    )

    print("=== INCIDENT PAYLOAD SCHEMA TEST PASSED ===")
    print(payload.model_dump_json(indent=2))


if __name__ == "__main__":
    test_schemas()
