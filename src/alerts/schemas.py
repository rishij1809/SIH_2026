"""
Pydantic Schemas and Models for Project Vigilance Stage 4 Alert System.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional
from uuid import uuid4
from pydantic import BaseModel, Field


class IncidentType(str, Enum):
    NORMAL = "normal"
    HELP_SIGNAL = "distress_signal"
    AGGRESSION = "aggression"
    FALL = "fall"
    INTRUSION = "intrusion"


class SpatialCoordinates(BaseModel):
    """Bounding box spatial coordinates."""
    x: float = Field(..., description="Top-left X or normalized X")
    y: float = Field(..., description="Top-left Y or normalized Y")
    w: float = Field(..., description="Bounding box width")
    h: float = Field(..., description="Bounding box height")


class IncidentPayload(BaseModel):
    """Structured Pydantic Alert Payload sent to C2 Command & Telephony dispatchers."""
    incident_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    camera_id: str = Field(..., example="CAM_ALPHA_GATE_03")
    location_tag: str = Field(default="BOP-Alpha-Outpost", example="North Sector Gate 3")
    event_type: IncidentType = Field(..., example=IncidentType.HELP_SIGNAL)
    confidence_score: float = Field(..., ge=0.0, le=1.0, example=0.92)
    track_id: int = Field(..., example=42)
    clip_path: Optional[str] = Field(default=None, example="data/incidents/clips/incident_123.mp4")
    clip_url: Optional[str] = Field(default=None, example="https://c2.surveillance.gov/clips/incident_123.mp4")
    keyframe_path: Optional[str] = Field(default=None, example="data/incidents/keyframes/keyframe_123.jpg")
    keyframe_url: Optional[str] = Field(default=None, example="https://c2.surveillance.gov/keyframes/keyframe_123.jpg")
    spatial_coordinates: Optional[SpatialCoordinates] = None
    extra_metadata: Dict[str, str] = Field(default_factory=dict)
