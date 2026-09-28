"""
Heuristic Verification & Temporal Gating Engine for Project Vigilance.
Enforces temporal persistence (M out of N positive detections), track/camera cooldown debouncing,
and Region-of-Interest (ROI) spatial boundary checking to eliminate false positives.
"""

from collections import deque
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.alerts.schemas import IncidentType
from src.utils.logger import setup_logger

logger = setup_logger("VerificationEngine")


class HeuristicVerificationEngine:
    """Eliminates false positive alerts via temporal persistence, spatial ROI filtering, and cooldown debouncing."""

    def __init__(
        self,
        confidence_threshold: float = 0.65,
        persistence_window_N: int = 5,
        persistence_threshold_M: int = 3,
        cooldown_sec: float = 60.0,
        enable_roi_filter: bool = False,
        roi_polygons: Optional[Dict[str, List[List[int]]]] = None
    ) -> None:
        """
        Args:
            confidence_threshold: Threshold theta for valid distress detection.
            persistence_window_N: Total recent temporal sliding window evaluations N.
            persistence_threshold_M: Required number of positive detections M out of N.
            cooldown_sec: Cooldown suppression period in seconds after alert trigger.
            enable_roi_filter: Whether to enforce spatial ROI polygon boundary checking.
            roi_polygons: Dictionary mapping camera_id -> polygon vertices [[x1, y1], [x2, y2], ...].
        """
        self.confidence_threshold = confidence_threshold
        self.persistence_window_N = persistence_window_N
        self.persistence_threshold_M = persistence_threshold_M
        self.cooldown_sec = cooldown_sec
        self.enable_roi_filter = enable_roi_filter
        self.roi_polygons = roi_polygons or {}

        # Rolling detection history per track: {track_id: deque([(timestamp, gesture_id, conf), ...], maxlen=N)}
        self.track_history: Dict[int, deque] = {}

        # Cooldown timestamps: {camera_id_track_id: last_alert_time}
        self.alert_cooldowns: Dict[str, float] = {}

    def is_in_roi(self, camera_id: str, bbox: List[float]) -> bool:
        """Checks if center of bounding box is inside camera's active ROI polygon."""
        if not self.enable_roi_filter or camera_id not in self.roi_polygons:
            return True

        poly_pts = np.array(self.roi_polygons[camera_id], dtype=np.int32)
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        res = cv2.pointPolygonTest(poly_pts, (center_x, center_y), False)
        return res >= 0  # True if inside or on boundary

    def evaluate_detection(
        self,
        camera_id: str,
        track_id: int,
        gesture_id: int,
        confidence: float,
        bbox: List[float],
        timestamp: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Evaluates whether a distress gesture detection qualifies as a verified incident trigger.

        Returns:
            Tuple of (is_verified: bool, reason: str)
        """
        now = timestamp or time.time()
        cooldown_key = f"{camera_id}_{track_id}"

        # 1. Normal behavior check
        if gesture_id == 0:
            return False, "Normal gesture"

        # 2. Confidence threshold check (theta)
        if confidence < self.confidence_threshold:
            return False, f"Confidence {confidence:.2f} < threshold {self.confidence_threshold}"

        # 3. Spatial ROI boundary check
        if not self.is_in_roi(camera_id, bbox):
            return False, f"Track {track_id} outside designated surveillance ROI zone"

        # 4. Cooldown debounce check
        if cooldown_key in self.alert_cooldowns:
            elapsed = now - self.alert_cooldowns[cooldown_key]
            if elapsed < self.cooldown_sec:
                return False, f"Track {track_id} in active cooldown ({elapsed:.1f}s / {self.cooldown_sec}s)"

        # 5. Temporal persistence check (M out of N)
        if track_id not in self.track_history:
            self.track_history[track_id] = deque(maxlen=self.persistence_window_N)

        self.track_history[track_id].append((now, gesture_id, confidence))

        history = self.track_history[track_id]
        if len(history) < self.persistence_threshold_M:
            return False, f"Insufficient temporal history ({len(history)}/{self.persistence_threshold_M})"

        # Count positive distress detections in window
        positive_count = sum(1 for _, g_id, c in history if g_id > 0 and c >= self.confidence_threshold)

        if positive_count >= self.persistence_threshold_M:
            # Verified trigger! Update cooldown timestamp
            self.alert_cooldowns[cooldown_key] = now
            logger.info(
                f"[VERIFIED INCIDENT] on [{camera_id}] Track #{track_id}: "
                f"Persistence {positive_count}/{len(history)} positive detections (Conf: {confidence:.2f})"
            )
            return True, "Verified persistence threshold satisfied"

        return False, f"Persistence count {positive_count}/{len(history)} < required {self.persistence_threshold_M}"

    def purge_old_tracks(self, max_idle_sec: float = 120.0) -> None:
        """Cleans up inactive track history and expired cooldown keys."""
        now = time.time()
        expired_cooldowns = [k for k, last_t in self.alert_cooldowns.items() if (now - last_t) > max_idle_sec]
        for k in expired_cooldowns:
            del self.alert_cooldowns[k]
