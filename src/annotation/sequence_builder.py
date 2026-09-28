"""
In-House Sequence Dataset Builder for Project Vigilance.
Extracts skeletal joint coordinate sequences from video frames/crops over sliding temporal windows.
Outputs indexed NumPy arrays and HDF5 files of shape (N_samples, Window_Size, Num_Keypoints, Channels).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import h5py
import numpy as np
from tqdm import tqdm

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

from src.utils.logger import setup_logger

logger = setup_logger("SequenceDatasetBuilder")


class SequenceDatasetBuilder:
    """Builds sliding-window skeletal keypoint sequence datasets for ST-GCN / Bi-LSTM models."""

    LABEL_MAP = {
        0: "Normal",
        1: "Universal Help Signal",
        2: "Aggression/Struggle",
        3: "Fall"
    }

    def __init__(
        self,
        window_size: int = 30,
        step_size: int = 5,
        num_keypoints: int = 33,
        channels: int = 3,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ) -> None:
        """
        Args:
            window_size: Temporal length of sliding window (number of frames per sample).
            step_size: Stride step between successive windows.
            num_keypoints: Number of skeletal joints (33 for MediaPipe Pose).
            channels: Channels per joint (3 for [x, y, visibility]).
            min_detection_confidence: Detection confidence for MediaPipe Pose.
            min_tracking_confidence: Tracking confidence for MediaPipe Pose.
        """
        self.window_size = window_size
        self.step_size = step_size
        self.num_keypoints = num_keypoints
        self.channels = channels
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        if HAS_MEDIAPIPE and hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            self.mp_pose = mp.solutions.pose
            self.pose_estimator = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
        else:
            logger.warning("MediaPipe pose solutions not present in current python environment. Frame pose extraction will use keypoint generator fallback.")
            self.mp_pose = None
            self.pose_estimator = None

    def extract_frame_keypoints(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Extracts 2D/3D keypoint matrix of shape (Num_Keypoints, Channels) from a single frame.
        """
        kp_matrix = np.zeros((self.num_keypoints, self.channels), dtype=np.float32)

        if self.pose_estimator is not None:
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self.pose_estimator.process(rgb_frame)

            if results.pose_landmarks:
                for idx, lm in enumerate(results.pose_landmarks.landmark):
                    if idx >= self.num_keypoints:
                        break
                    if self.channels == 3:
                        kp_matrix[idx] = [lm.x, lm.y, lm.visibility]
                    else:
                        kp_matrix[idx] = [lm.x, lm.y]
        return kp_matrix

    def extract_sequence_from_video(
        self,
        video_path: Union[str, Path],
        label_id: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts sliding window keypoint sequences from a video file.

        Returns:
            X_seqs: Array of shape (N_samples, Window_Size, Num_Keypoints, Channels)
            y_seqs: Array of shape (N_samples,) containing integer class labels
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video file: {video_path}")
            return np.empty((0, self.window_size, self.num_keypoints, self.channels)), np.empty((0,), dtype=int)

        raw_keypoints_list = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            kp = self.extract_frame_keypoints(frame)
            raw_keypoints_list.append(kp)

        cap.release()

        n_frames = len(raw_keypoints_list)
        if n_frames < self.window_size:
            logger.warning(
                f"Video {Path(video_path).name} has only {n_frames} frames (< window_size {self.window_size}). Skipping."
            )
            return np.empty((0, self.window_size, self.num_keypoints, self.channels)), np.empty((0,), dtype=int)

        # Sliding window sequence creation
        sequences = []
        labels = []
        for start_idx in range(0, n_frames - self.window_size + 1, self.step_size):
            window = raw_keypoints_list[start_idx : start_idx + self.window_size]
            sequences.append(np.array(window, dtype=np.float32))
            labels.append(label_id)

        X_array = np.array(sequences, dtype=np.float32)
        y_array = np.array(labels, dtype=np.int64)

        return X_array, y_array

    def generate_synthetic_demo_sequences(
        self,
        num_samples_per_class: int = 25
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates realistic synthetic skeletal keypoint sequences for rapid prototyping & pipeline validation.
        """
        logger.info(f"Generating synthetic keypoint sequences ({num_samples_per_class} per class)...")
        all_X = []
        all_y = []

        np.random.seed(42)

        for label_id in self.LABEL_MAP.keys():
            for _ in range(num_samples_per_class):
                # Base posture keypoint template (normalized 0..1 coordinates)
                base_pose = np.random.uniform(0.3, 0.7, size=(self.num_keypoints, self.channels)).astype(np.float32)
                if self.channels == 3:
                    base_pose[:, 2] = np.random.uniform(0.7, 1.0, size=(self.num_keypoints,))  # High visibility

                seq = np.zeros((self.window_size, self.num_keypoints, self.channels), dtype=np.float32)

                # Simulate gesture class characteristics
                for t in range(self.window_size):
                    t_norm = t / float(self.window_size)
                    noise = np.random.normal(0, 0.01, size=(self.num_keypoints, self.channels))

                    if label_id == 0:  # Normal standing/walking motion
                        offset = np.sin(t_norm * 2 * np.pi) * 0.02
                        frame_kp = base_pose + offset + noise

                    elif label_id == 1:  # Universal Help Signal (hands raised high above head)
                        frame_kp = base_pose.copy()
                        # Wrist joints (15: left wrist, 16: right wrist) pulled high (y close to 0)
                        frame_kp[15, 1] = 0.15 + np.sin(t_norm * 4 * np.pi) * 0.03
                        frame_kp[16, 1] = 0.15 + np.cos(t_norm * 4 * np.pi) * 0.03
                        frame_kp += noise

                    elif label_id == 2:  # Aggression/Struggle (high dynamic jitter across limbs)
                        frame_kp = base_pose + np.random.normal(0, 0.05, size=(self.num_keypoints, self.channels))

                    elif label_id == 3:  # Fall (rapid downward drop y -> 0.85, rotation)
                        fall_progress = min(1.0, t_norm * 1.5)
                        frame_kp = base_pose.copy()
                        frame_kp[:, 1] += fall_progress * 0.4  # Drop downward
                        frame_kp += noise

                    seq[t] = np.clip(frame_kp, 0.0, 1.0)

                all_X.append(seq)
                all_y.append(label_id)

        X = np.array(all_X, dtype=np.float32)
        y = np.array(all_y, dtype=np.int64)

        # Shuffle
        idx = np.arange(len(y))
        np.random.shuffle(idx)
        return X[idx], y[idx]

    def save_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        output_dir: Union[str, Path],
        h5_filename: str = "sequences.h5",
        npy_x_name: str = "X_sequences.npy",
        npy_y_name: str = "y_sequences.npy"
    ) -> Dict[str, str]:
        """
        Saves structured keypoint sequence tensor to NumPy .npy files and HDF5 archive.

        Expected X shape: (N_samples, Window_Size, Num_Keypoints, Channels)
        Expected y shape: (N_samples,)
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        npy_x_path = out_path / npy_x_name
        npy_y_path = out_path / npy_y_name
        h5_path = out_path / h5_filename

        # Save NumPy binaries
        np.save(npy_x_path, X)
        np.save(npy_y_path, y)
        logger.info(f"Saved NumPy sequences X: {X.shape} -> {npy_x_path}")
        logger.info(f"Saved NumPy labels y: {y.shape} -> {npy_y_path}")

        # Save HDF5 archive with compression & label attributes
        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset("X", data=X, compression="gzip", compression_opts=4)
            hf.create_dataset("y", data=y, compression="gzip")
            
            # Store metadata attributes
            hf.attrs["window_size"] = self.window_size
            hf.attrs["num_keypoints"] = self.num_keypoints
            hf.attrs["channels"] = self.channels
            hf.attrs["num_samples"] = X.shape[0]
            for k, v in self.LABEL_MAP.items():
                hf.attrs[f"label_{k}"] = v

        logger.info(f"Saved HDF5 dataset archive -> {h5_path}")

        return {
            "npy_x": str(npy_x_path),
            "npy_y": str(npy_y_path),
            "h5": str(h5_path)
        }
