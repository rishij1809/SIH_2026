"""
Multi-Object Tracking & Tracklet Buffer Manager for Project Vigilance.
Assigns persistent Track IDs to detected persons across video frames,
maintains sliding-window temporal keypoint FIFO buffers, and purges stale tracklets.
"""

from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

from src.utils.logger import setup_logger

logger = setup_logger("TrackletManager")


class SimpleIoUTracker:
    """Lightweight IoU & Centroid Multi-Object Tracker for real-time person tracking."""

    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 30) -> None:
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.next_track_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}

    @staticmethod
    def compute_iou(boxA: List[float], boxB: List[float]) -> float:
        """Computes IoU between two bounding boxes [x1, y1, x2, y2]."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
        boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def update(self, detections: List[List[float]], scores: List[float]) -> Dict[int, List[float]]:
        """
        Updates tracked targets with new detection bounding boxes [x1, y1, x2, y2].
        Returns dictionary mapping active track_id to current bounding box [x1, y1, x2, y2].
        """
        active_track_boxes = {}
        matched_det_indices = set()

        # Attempt to match existing tracks with new detections
        for track_id, track_info in list(self.tracks.items()):
            last_box = track_info["box"]
            best_iou = 0.0
            best_det_idx = -1

            for det_idx, det_box in enumerate(detections):
                if det_idx in matched_det_indices:
                    continue
                iou = self.compute_iou(last_box, det_box)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx

            if best_iou >= self.iou_threshold and best_det_idx != -1:
                # Match found! Update track
                matched_det_indices.add(best_det_idx)
                self.tracks[track_id]["box"] = detections[best_det_idx]
                self.tracks[track_id]["score"] = scores[best_det_idx]
                self.tracks[track_id]["lost_frames"] = 0
                active_track_boxes[track_id] = detections[best_det_idx]
            else:
                # Increment lost frames counter
                self.tracks[track_id]["lost_frames"] += 1
                if self.tracks[track_id]["lost_frames"] > self.max_lost_frames:
                    # Purge stale track
                    del self.tracks[track_id]

        # Register new detections as new track IDs
        for det_idx, det_box in enumerate(detections):
            if det_idx not in matched_det_indices:
                t_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[t_id] = {
                    "box": det_box,
                    "score": scores[det_idx],
                    "lost_frames": 0
                }
                active_track_boxes[t_id] = det_box

        return active_track_boxes


class TrackletManager:
    """Manages multi-object tracking & rolling FIFO keypoint sequence buffers per track ID."""

    def __init__(
        self,
        window_length: int = 30,
        num_keypoints: int = 33,
        channels: int = 3,
        max_stale_frames: int = 30,
        iou_threshold: float = 0.3
    ) -> None:
        """
        Args:
            window_length: Temporal window length (number of frames per sample).
            num_keypoints: Number of body joints (33 for MediaPipe Pose).
            channels: Channels per keypoint (3 for [x, y, confidence]).
            max_stale_frames: Inactive frame limit before purging tracklet memory.
            iou_threshold: IoU matching threshold for track association.
        """
        self.window_length = window_length
        self.num_keypoints = num_keypoints
        self.channels = channels
        self.max_stale_frames = max_stale_frames

        # Multi-Object Tracker
        self.tracker = SimpleIoUTracker(iou_threshold=iou_threshold, max_lost_frames=max_stale_frames)

        # Tracklet Keypoint FIFO Buffers: {track_id: deque of shape (window_length, V, C)}
        self.keypoint_buffers: Dict[int, deque] = {}

        # MediaPipe Pose Estimator
        if HAS_MEDIAPIPE:
            self.mp_pose = mp.solutions.pose
            self.pose_estimator = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,  # Ultra-fast mode
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4
            )
        else:
            self.mp_pose = None
            self.pose_estimator = None

    def extract_crop_keypoints(self, person_crop: np.ndarray) -> np.ndarray:
        """Extracts (num_keypoints, channels) matrix from person image crop."""
        kp_matrix = np.zeros((self.num_keypoints, self.channels), dtype=np.float32)

        if self.pose_estimator is not None and person_crop.size > 0:
            rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            results = self.pose_estimator.process(rgb_crop)

            if results.pose_landmarks:
                for idx, lm in enumerate(results.pose_landmarks.landmark):
                    if idx >= self.num_keypoints:
                        break
                    if self.channels == 3:
                        kp_matrix[idx] = [lm.x, lm.y, lm.visibility]
                    else:
                        kp_matrix[idx] = [lm.x, lm.y]
        return kp_matrix

    def update_frame(
        self,
        frame_bgr: np.ndarray,
        detections: List[List[float]],
        scores: List[float]
    ) -> Dict[int, Tuple[List[float], Optional[np.ndarray]]]:
        """
        Updates multi-object tracks and appends newly extracted keypoints to tracklet buffers.

        Args:
            frame_bgr: Current video frame array.
            detections: List of bounding boxes [x1, y1, x2, y2] in pixel coordinates.
            scores: Bounding box detection confidence scores.

        Returns:
            Dictionary mapping track_id -> (bbox [x1, y1, x2, y2], sequence_tensor (30, V, C) or None).
        """
        H, W = frame_bgr.shape[:2]
        active_tracks = self.tracker.update(detections, scores)

        result_tracks = {}

        # Update keypoint sequence buffers for active tracks
        for track_id, bbox in active_tracks.items():
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)

            person_crop = frame_bgr[y1:y2, x1:x2]
            keypoints = self.extract_crop_keypoints(person_crop)

            if track_id not in self.keypoint_buffers:
                self.keypoint_buffers[track_id] = deque(maxlen=self.window_length)

            self.keypoint_buffers[track_id].append(keypoints)

            # Check if buffer has reached full temporal window size
            seq_array = None
            if len(self.keypoint_buffers[track_id]) == self.window_length:
                seq_array = np.array(self.keypoint_buffers[track_id], dtype=np.float32)

            result_tracks[track_id] = (bbox, seq_array)

        # Purge keypoint buffers for stale/deleted track IDs
        active_set = set(active_tracks.keys())
        stale_ids = [t_id for t_id in self.keypoint_buffers if t_id not in active_set]
        for t_id in stale_ids:
            del self.keypoint_buffers[t_id]

        return result_tracks
