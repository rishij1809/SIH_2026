#!/usr/bin/env python3
"""
Tactical Command & Control (C2) Backend Server for Project Vigilance / IBVAP (`c2_backend.py`).
Built with FastAPI, WebSockets, and Pydantic validation.
Conforms strictly to C4ISR Tactical Web Dashboard API Contracts.
"""

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import psutil

from src.utils.logger import setup_logger

logger = setup_logger("C2Backend")

app = FastAPI(
    title="Project Vigilance / IBVAP - Tactical C2 Backend API",
    description="Real-time Command & Control REST and WebSocket API for Safety Surveillance",
    version="3.0.0"
)

# Enable CORS for React frontend (Vite port 5173 / 3000 / 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-Memory Databases
CAMERAS_DB: Dict[str, Dict[str, Any]] = {
    "CAM-01": {
        "id": "CAM-01",
        "name": "DEVICE-LOCAL-WEBCAM",
        "location": "Sector 4 Main Entrance (Device Camera)",
        "rtsp_stream_url": "webcam",
        "status": "ONLINE",
        "fps": 30.0,
        "active_alerts_count": 0
    },
    "CAM-02": {
        "id": "CAM-02",
        "name": "EAST-PERIMETER-WALL",
        "location": "East Boundary Fence 12",
        "rtsp_stream_url": "data/raw_videos/demo_cctv_feed.mp4",
        "status": "ONLINE",
        "fps": 12.0,
        "active_alerts_count": 0
    },
    "CAM-03": {
        "id": "CAM-03",
        "name": "SOUTH-CORRIDOR-CHECK",
        "location": "South Patrol Route A",
        "rtsp_stream_url": "data/raw_videos/demo_cctv_feed.mp4",
        "status": "ONLINE",
        "fps": 12.0,
        "active_alerts_count": 0
    },
    "CAM-04": {
        "id": "CAM-04",
        "name": "VEHICLE-INSPECTION-BAY",
        "location": "Outpost Logistics Gate",
        "rtsp_stream_url": "data/raw_videos/demo_cctv_feed.mp4",
        "status": "ONLINE",
        "fps": 12.0,
        "active_alerts_count": 0
    },
    "CAM-05": {
        "id": "CAM-05",
        "name": "TOWER-NORTH-OPTICS",
        "location": "Observation Tower 1",
        "rtsp_stream_url": "data/raw_videos/demo_cctv_feed.mp4",
        "status": "DEGRADED",
        "fps": 8.5,
        "active_alerts_count": 0
    },
    "CAM-06": {
        "id": "CAM-06",
        "name": "ARMORY-PERIMETER-S",
        "location": "Tactical Depot South",
        "rtsp_stream_url": "data/raw_videos/demo_cctv_feed.mp4",
        "status": "ONLINE",
        "fps": 12.0,
        "active_alerts_count": 0
    },
    "CAM-07": {
        "id": "CAM-07",
        "name": "HELI-PAD-WEST-APPROACH",
        "location": "Airfield West Boundary",
        "rtsp_stream_url": "data/raw_videos/demo_cctv_feed.mp4",
        "status": "ONLINE",
        "fps": 12.0,
        "active_alerts_count": 0
    },
    "CAM-08": {
        "id": "CAM-08",
        "name": "COMMAND-BUNKER-MAIN",
        "location": "HQ Internal Corridor",
        "rtsp_stream_url": "data/raw_videos/demo_cctv_feed.mp4",
        "status": "ONLINE",
        "fps": 12.0,
        "active_alerts_count": 0
    }
}

INCIDENTS_DB: Dict[str, Dict[str, Any]] = {
    "INC-8091": {
        "id": "INC-8091",
        "camera_id": "CAM-01",
        "camera_name": "BOP-ALPHA-GATE-01",
        "location_tag": "Sector 4 North Gate",
        "event_type": "distress_signal",
        "confidence_score": 0.948,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "clip_url": "/media/clips/demo_clip.mp4",
        "keyframe_url": "/media/keyframes/demo_keyframe.jpg",
        "status": "UNACKNOWLEDGED",
        "operator_notes": "",
        "track_id": 99,
        "spatial_coordinates": {"x": 250.0, "y": 180.0, "w": 130.0, "h": 300.0}
    },
    "INC-8090": {
        "id": "INC-8090",
        "camera_id": "CAM-02",
        "camera_name": "EAST-PERIMETER-WALL",
        "location_tag": "East Boundary Fence 12",
        "event_type": "fall",
        "confidence_score": 0.884,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "clip_url": "/media/clips/demo_clip.mp4",
        "keyframe_url": "/media/keyframes/demo_keyframe.jpg",
        "status": "ACKNOWLEDGED",
        "operator_notes": "Patrol dispatched to assist fallen personnel.",
        "track_id": 45,
        "spatial_coordinates": {"x": 100.0, "y": 200.0, "w": 150.0, "h": 120.0}
    }
}


# WebSocket Manager
class WebSocketManager:
    def __init__(self) -> None:
        self.alert_subscribers: List[WebSocket] = []
        self.telemetry_subscribers: List[WebSocket] = []

    async def connect_alert(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.alert_subscribers.append(websocket)

    def disconnect_alert(self, websocket: WebSocket) -> None:
        if websocket in self.alert_subscribers:
            self.alert_subscribers.remove(websocket)

    async def broadcast_alert(self, payload: Dict[str, Any]) -> None:
        for ws in list(self.alert_subscribers):
            try:
                await ws.send_json(payload)
            except Exception:
                self.disconnect_alert(ws)

    async def connect_telemetry(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.telemetry_subscribers.append(websocket)

    def disconnect_telemetry(self, websocket: WebSocket) -> None:
        if websocket in self.telemetry_subscribers:
            self.telemetry_subscribers.remove(websocket)

    async def broadcast_telemetry(self, data: Dict[str, Any]) -> None:
        for ws in list(self.telemetry_subscribers):
            try:
                await ws.send_json(data)
            except Exception:
                self.disconnect_telemetry(ws)


ws_manager = WebSocketManager()


# Request Pydantic Schemas
class TriageRequest(BaseModel):
    status: str = Field(..., example="ACKNOWLEDGED")  # "ACKNOWLEDGED" | "DISMISSED"
    operator_notes: Optional[str] = Field(default="", example="Verified emergency. Tactical response en route.")


# REST API Endpoints
@app.get("/api/v1/cameras")
async def get_cameras():
    """Returns list of active camera streams and health metrics."""
    return list(CAMERAS_DB.values())


@app.get("/api/v1/incidents")
async def get_incidents(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    status: Optional[str] = None,
    event_type: Optional[str] = None,
    camera_id: Optional[str] = None
):
    """Returns paginated list of past incidents with filtering."""
    incidents = list(INCIDENTS_DB.values())

    if status:
        incidents = [i for i in incidents if i.get("status") == status]
    if event_type:
        incidents = [i for i in incidents if i.get("event_type") == event_type]
    if camera_id:
        incidents = [i for i in incidents if i.get("camera_id") == camera_id]

    # Sort descending by timestamp
    incidents.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    return incidents[start_idx:end_idx]


@app.patch("/api/v1/incidents/{incident_id}/triage")
async def triage_incident(incident_id: str, req: TriageRequest):
    """Operator endpoint to mark an incident as ACKNOWLEDGED or DISMISSED with notes."""
    if incident_id not in INCIDENTS_DB:
        raise HTTPException(status_code=404, detail=f"Incident '{incident_id}' not found")

    INCIDENTS_DB[incident_id]["status"] = req.status
    INCIDENTS_DB[incident_id]["operator_notes"] = req.operator_notes
    INCIDENTS_DB[incident_id]["triaged_at"] = datetime.now(timezone.utc).isoformat()

    # Decrement active alert count if camera match
    cam_id = INCIDENTS_DB[incident_id].get("camera_id")
    if cam_id in CAMERAS_DB and CAMERAS_DB[cam_id]["active_alerts_count"] > 0:
        CAMERAS_DB[cam_id]["active_alerts_count"] -= 1

    logger.info(f"Operator triaged incident [{incident_id}] -> {req.status}")
    return INCIDENTS_DB[incident_id]


@app.get("/api/v1/telemetry")
async def get_telemetry():
    """Returns real-time edge node system vitals."""
    return {
        "edge_node_id": "BOP-ALPHA-NODE-01",
        "gpu_temp_c": 44.5 + round(random.uniform(-1.5, 2.5), 1),
        "gpu_utilization_pct": round(random.uniform(28.0, 42.0), 1),
        "vram_used_mb": 2480,
        "vram_total_mb": 8192,
        "pipeline_latency_ms": round(random.uniform(12.8, 15.4), 1),
        "disk_usage_pct": 74.2,
        "connectivity": "HEALTHY"
    }


class RealtimeAlertRequest(BaseModel):
    camera_id: str = "CAM-01"
    event_type: str = "universal_sos"  # "universal_sos" | "weapon_knife" | "fall"
    confidence_score: float = 0.95
    spatial_coordinates: Optional[Dict[str, float]] = None


@app.post("/api/v1/alerts/trigger")
async def trigger_realtime_alert(req: RealtimeAlertRequest):
    """Triggers verified real-time threat alert (Universal SOS Hand Sign / Knife Weapon)."""
    inc_id = f"INC-{random.randint(9000, 9999)}"
    cam_info = CAMERAS_DB.get(req.camera_id, CAMERAS_DB["CAM-01"])

    new_incident = {
        "id": inc_id,
        "camera_id": req.camera_id,
        "camera_name": cam_info["name"],
        "location_tag": cam_info["location"],
        "event_type": req.event_type,
        "confidence_score": req.confidence_score,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "clip_url": "/media/clips/demo_clip.mp4",
        "keyframe_url": "/media/keyframes/demo_keyframe.jpg",
        "status": "UNACKNOWLEDGED",
        "operator_notes": f"Real-Time Detection: {req.event_type.upper()}",
        "track_id": random.randint(10, 99),
        "spatial_coordinates": req.spatial_coordinates or {"x": 200.0, "y": 150.0, "w": 140.0, "h": 280.0}
    }

    INCIDENTS_DB[inc_id] = new_incident
    CAMERAS_DB[req.camera_id]["active_alerts_count"] += 1

    await ws_manager.broadcast_alert(new_incident)
    logger.info(f"🚨 REALTIME THREAT BROADCAST: [{inc_id}] {req.event_type} (Conf: {req.confidence_score:.2f})")
    return new_incident


@app.post("/api/v1/alerts/simulate")
async def simulate_alert(
    camera_id: str = "CAM-01",
    event_type: str = "distress_signal",
    confidence: float = 0.948
):
    """Generates synthetic distress incident and broadcasts over WebSocket."""
    inc_id = f"INC-{random.randint(8100, 9999)}"
    cam_info = CAMERAS_DB.get(camera_id, CAMERAS_DB["CAM-01"])

    new_incident = {
        "id": inc_id,
        "camera_id": camera_id,
        "camera_name": cam_info["name"],
        "location_tag": cam_info["location"],
        "event_type": event_type,
        "confidence_score": confidence,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "clip_url": "/media/clips/demo_clip.mp4",
        "keyframe_url": "/media/keyframes/demo_keyframe.jpg",
        "status": "UNACKNOWLEDGED",
        "operator_notes": "",
        "track_id": random.randint(10, 99),
        "spatial_coordinates": {"x": 200.0, "y": 150.0, "w": 140.0, "h": 280.0}
    }

    INCIDENTS_DB[inc_id] = new_incident
    CAMERAS_DB[camera_id]["active_alerts_count"] += 1

    await ws_manager.broadcast_alert(new_incident)
    return new_incident


# WebSockets
@app.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket):
    await ws_manager.connect_alert(websocket)
    try:
        while True:
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect_alert(websocket)


@app.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket):
    await ws_manager.connect_telemetry(websocket)
    try:
        while True:
            await asyncio.sleep(2.0)
            telemetry = await get_telemetry()
            await ws_manager.broadcast_telemetry(telemetry)
    except WebSocketDisconnect:
        ws_manager.disconnect_telemetry(websocket)


# Mount Static Media Storage
incidents_dir = Path("data/incidents")
incidents_dir.mkdir(parents=True, exist_ok=True)
(incidents_dir / "clips").mkdir(exist_ok=True)
(incidents_dir / "keyframes").mkdir(exist_ok=True)

app.mount("/media", StaticFiles(directory=str(incidents_dir)), name="media")

if __name__ == "__main__":
    import uvicorn
    logger.info("Launching Project Vigilance / IBVAP FastAPI Backend on http://0.0.0.0:8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
