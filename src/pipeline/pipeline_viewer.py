"""
Visual Debugger & Performance Telemetry Viewer for Project Vigilance.
Renders real-time HUD overlays, bounding box status labels, skeletal keypoints, and distress alerts.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


class PipelineTelemetryViewer:
    """Renders visual bounding box overlays, distress status tags, and real-time HUD metrics."""

    COLOR_NORMAL = (0, 255, 120)        # Green
    COLOR_HELP_SIGNAL = (0, 215, 255)   # Amber / Yellow
    COLOR_AGGRESSION = (0, 0, 255)      # Bright Red
    COLOR_FALL = (0, 0, 255)            # Bright Red

    COLOR_MAP = {
        0: COLOR_NORMAL,
        1: COLOR_HELP_SIGNAL,
        2: COLOR_AGGRESSION,
        3: COLOR_FALL
    }

    def __init__(self, window_name: str = "Project Vigilance - Safety Surveillance Telemetry") -> None:
        self.window_name = window_name

    def draw_hud(
        self,
        frame: np.ndarray,
        ingestion_fps: float,
        inference_fps: float,
        latency_ms: float,
        active_tracks_count: int,
        alert_count: int
    ) -> np.ndarray:
        """Renders HUD telemetry header overlay on frame."""
        overlay = frame.copy()
        H, W = frame.shape[:2]

        # Draw semi-transparent header bar
        header_height = 45
        cv2.rectangle(overlay, (0, 0), (W, header_height), (15, 15, 15), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Telemetry text string
        hud_text = (
            f"VIGILANCE TELEMETRY | Ingest: {ingestion_fps:.1f} FPS | "
            f"Infer: {inference_fps:.1f} FPS | Latency: {latency_ms:.1f} ms | "
            f"Active Persons: {active_tracks_count} | Alerts: {alert_count}"
        )
        
        cv2.putText(
            frame,
            hud_text,
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        return frame

    def render_frame(
        self,
        frame_bgr: np.ndarray,
        pipeline_output: Dict[str, Any],
        ingestion_fps: float = 0.0
    ) -> np.ndarray:
        """
        Draws tracking bounding boxes, gesture classification labels, alert badges, and telemetry HUD.
        """
        out_frame = frame_bgr.copy()
        
        tracked_persons = pipeline_output.get("tracked_persons", [])
        alerts = pipeline_output.get("alerts", [])
        latency_ms = pipeline_output.get("latency_ms", 0.0)
        inference_fps = pipeline_output.get("fps", 0.0)

        # 1. Render Tracked Persons & Bounding Boxes
        for person in tracked_persons:
            track_id = person["track_id"]
            bbox = person["bbox"]
            gesture_id = person["gesture_id"]
            gesture_name = person["gesture_name"]
            conf = person["confidence"]
            is_alert = person["is_alert"]

            color = self.COLOR_MAP.get(gesture_id, self.COLOR_NORMAL)
            x1, y1, x2, y2 = map(int, bbox)

            # Draw bounding box (thicker if alert)
            thickness = 3 if is_alert else 2
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, thickness)

            # Label banner
            label_text = f"ID #{track_id} | {gesture_name}"
            if conf > 0 and gesture_id > 0:
                label_text += f" ({conf*100:.0f}%)"

            # Text background pill
            (txt_w, txt_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(out_frame, (x1, y1 - txt_h - 10), (x1 + txt_w + 10, y1), color, -1)
            cv2.putText(out_frame, label_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Flash alert banner if distress detected
            if is_alert:
                cv2.rectangle(out_frame, (x1, y2 + 5), (x2, y2 + 25), (0, 0, 255), -1)
                cv2.putText(out_frame, "!!! DISTRESS ALERT !!!", (x1 + 5, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 2. Render Telemetry HUD
        out_frame = self.draw_hud(
            out_frame,
            ingestion_fps=ingestion_fps,
            inference_fps=inference_fps,
            latency_ms=latency_ms,
            active_tracks_count=len(tracked_persons),
            alert_count=len(alerts)
        )

        return out_frame

    def show(self, frame: np.ndarray) -> bool:
        """
        Displays frame in OpenCV UI window. Returns False if user presses 'q' or ESC to exit.
        """
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord('q'), ord('Q')]:
            return False
        return True

    def close(self) -> None:
        """Destroys OpenCV display window."""
        cv2.destroyAllWindows()
