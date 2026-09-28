"""
Rolling Circular Video Buffer & Incident Clip Packager for Project Vigilance.
Maintains in-memory pre-incident ring buffer, captures post-incident frames, and exports MP4 video clips & annotated keyframes.
"""

from collections import deque
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
from uuid import uuid4
import cv2
import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger("IncidentClipRecorder")


class IncidentClipRecorder:
    """Manages pre/post incident circular video ring-buffers and encodes MP4 clips & JPEGs."""

    def __init__(
        self,
        storage_dir: Union[str, Path] = "data/incidents",
        pre_incident_sec: float = 5.0,
        post_incident_sec: float = 5.0,
        fps: float = 12.0
    ) -> None:
        """
        Args:
            storage_dir: Root output directory for incident clips and keyframes.
            pre_incident_sec: Pre-trigger buffer duration in seconds.
            post_incident_sec: Post-trigger recording duration in seconds.
            fps: Video recording FPS.
        """
        self.storage_dir = Path(storage_dir)
        self.clips_dir = self.storage_dir / "clips"
        self.keyframes_dir = self.storage_dir / "keyframes"

        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.keyframes_dir.mkdir(parents=True, exist_ok=True)

        self.pre_incident_sec = pre_incident_sec
        self.post_incident_sec = post_incident_sec
        self.fps = fps

        self.pre_buffer_maxlen = max(1, int(round(pre_incident_sec * fps)))
        self.post_buffer_maxlen = max(1, int(round(post_incident_sec * fps)))

        # Pre-incident ring buffer per camera: {camera_id: deque([(timestamp, frame_bgr), ...], maxlen=N)}
        self.pre_ring_buffers: Dict[str, deque] = {}

    def push_frame(self, camera_id: str, frame_bgr: np.ndarray, timestamp: Optional[float] = None) -> None:
        """Pushes incoming stream frame into camera's pre-incident ring buffer."""
        if camera_id not in self.ring_buffers_dict:
            self.pre_ring_buffers[camera_id] = deque(maxlen=self.pre_buffer_maxlen)

        ts = timestamp or time.time()
        self.pre_ring_buffers[camera_id].append((ts, frame_bgr.copy()))

    @property
    def ring_buffers_dict(self) -> Dict[str, deque]:
        return self.pre_ring_buffers

    def save_annotated_keyframe(
        self,
        frame_bgr: np.ndarray,
        incident_id: str,
        gesture_name: str,
        confidence: float,
        bbox: List[float]
    ) -> Path:
        """Draws bounding box banner, keypoints, and timestamp burned-in overlay and saves keyframe JPEG."""
        annotated = frame_bgr.copy()
        H, W = annotated.shape[:2]

        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Burned-in banner
        banner_text = f"INCIDENT DETECTED: {gesture_name.upper()} ({confidence*100:.1f}%)"
        cv2.rectangle(annotated, (0, H - 40), (W, H), (0, 0, 0), -1)
        cv2.putText(annotated, banner_text, (20, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        out_path = self.keyframes_dir / f"keyframe_{incident_id}.jpg"
        cv2.imwrite(str(out_path), annotated)
        logger.info(f"Saved annotated keyframe -> {out_path}")
        return out_path

    def record_incident_clip(
        self,
        camera_id: str,
        incident_id: str,
        post_frames: List[np.ndarray],
        gesture_name: str = "Distress",
        confidence: float = 0.90,
        bbox: Optional[List[float]] = None
    ) -> Tuple[Path, Optional[Path]]:
        """
        Combines pre-incident ring buffer frames + post-incident frames and exports MP4 video clip.

        Returns:
            Tuple of (clip_path, keyframe_path)
        """
        pre_frames_data = list(self.pre_ring_buffers.get(camera_id, []))
        pre_frames = [f[1] for f in pre_frames_data]

        all_frames = pre_frames + post_frames
        if not all_frames:
            logger.warning(f"No frames available for incident {incident_id}. Creating fallback frame...")
            all_frames = [np.full((720, 1280, 3), 50, dtype=np.uint8)]

        sample_frame = all_frames[0]
        H, W = sample_frame.shape[:2]

        out_clip_path = self.clips_dir / f"incident_{incident_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(out_clip_path), fourcc, self.fps, (W, H))

        for frame in all_frames:
            out.write(frame)
        out.release()

        logger.info(f"Exported combined 10-second MP4 clip ({len(all_frames)} frames) -> {out_clip_path}")

        # Save peak-confidence keyframe
        keyframe_path = None
        if bbox is not None and len(all_frames) > 0:
            keyframe_path = self.save_annotated_keyframe(
                all_frames[len(pre_frames) // 2 if len(pre_frames) > 0 else 0],
                incident_id,
                gesture_name,
                confidence,
                bbox
            )

        return out_clip_path, keyframe_path
