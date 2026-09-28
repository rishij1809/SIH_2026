"""
Temporal & Spatial Keypoint Sequence Augmentor for Project Vigilance.
Performs spatial jitter, scaling, 2D rotation, frame dropouts, and temporal speed stretching on (T, V, C) tensors.
"""

from typing import Tuple
import numpy as np
from scipy.interpolate import interp1d

from src.utils.logger import setup_logger

logger = setup_logger("SequenceAugmentor")


class SequenceAugmentor:
    """Applies spatial and temporal augmentations to skeletal keypoint sequence matrices."""

    def __init__(
        self,
        jitter_std: float = 0.015,
        scale_range: Tuple[float, float] = (0.85, 1.15),
        rotation_deg_max: float = 10.0,
        frame_dropout_prob: float = 0.3,
        max_dropout_frames: int = 3,
        speed_stretch_prob: float = 0.3
    ) -> None:
        """
        Args:
            jitter_std: Standard deviation of spatial Gaussian noise.
            scale_range: Min and max scale multipliers.
            rotation_deg_max: Max 2D rotation angle in degrees.
            frame_dropout_prob: Probability of applying frame dropout.
            max_dropout_frames: Maximum number of frames to drop.
            speed_stretch_prob: Probability of temporal speed stretch/compression.
        """
        self.jitter_std = jitter_std
        self.scale_range = scale_range
        self.rotation_deg_max = rotation_deg_max
        self.frame_dropout_prob = frame_dropout_prob
        self.max_dropout_frames = max_dropout_frames
        self.speed_stretch_prob = speed_stretch_prob

    def add_spatial_jitter(self, sequence: np.ndarray) -> np.ndarray:
        """Adds zero-mean Gaussian noise to (x, y) keypoint coordinates."""
        aug_seq = sequence.copy()
        T, V, C = aug_seq.shape
        noise = np.random.normal(0, self.jitter_std, size=(T, V, 2)).astype(np.float32)
        aug_seq[:, :, :2] += noise
        aug_seq[:, :, :2] = np.clip(aug_seq[:, :, :2], 0.0, 1.0)
        return aug_seq

    def spatial_scale_and_rotate(self, sequence: np.ndarray) -> np.ndarray:
        """Applies spatial scaling and 2D rotation around center (0.5, 0.5)."""
        aug_seq = sequence.copy()
        T, V, C = aug_seq.shape

        # Random scaling
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])

        # Random rotation angle
        angle_rad = np.radians(np.random.uniform(-self.rotation_deg_max, self.rotation_deg_max))
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

        center = np.array([0.5, 0.5], dtype=np.float32)

        for t in range(T):
            coords = aug_seq[t, :, :2] - center
            coords = coords * scale
            coords = np.dot(coords, rot_mat.T)
            aug_seq[t, :, :2] = np.clip(coords + center, 0.0, 1.0)

        return aug_seq

    def apply_frame_dropout(self, sequence: np.ndarray) -> np.ndarray:
        """Randomly drops frames by zeroing keypoints or replacing with neighbor frames."""
        aug_seq = sequence.copy()
        T, V, C = aug_seq.shape
        
        if np.random.rand() > self.frame_dropout_prob:
            return aug_seq

        n_drop = np.random.randint(1, self.max_dropout_frames + 1)
        drop_indices = np.random.choice(range(1, T - 1), size=min(n_drop, T - 2), replace=False)

        for idx in drop_indices:
            # Interpolate between previous and next frame
            aug_seq[idx] = 0.5 * (aug_seq[idx - 1] + aug_seq[idx + 1])

        return aug_seq

    def temporal_speed_stretch(self, sequence: np.ndarray) -> np.ndarray:
        """Stretches or compresses sequence temporally and resamples to original window size T."""
        aug_seq = sequence.copy()
        T, V, C = aug_seq.shape

        if np.random.rand() > self.speed_stretch_prob:
            return aug_seq

        speed_factor = np.random.uniform(0.75, 1.25)
        new_len = int(round(T * speed_factor))
        if new_len < 5:
            return aug_seq

        orig_time = np.linspace(0, 1, T)
        stretched_time = np.linspace(0, 1, new_len)
        
        flat_seq = aug_seq.reshape(T, V * C)
        interp_fn = interp1d(orig_time, flat_seq, axis=0, kind="linear", fill_value="extrapolate")
        
        resampled_flat = interp_fn(np.linspace(0, 1, T))
        return resampled_flat.reshape(T, V, C).astype(np.float32)

    def augment_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Applies full spatial-temporal augmentation pipeline to sequence of shape (Window_Size, Num_Keypoints, Channels).
        """
        aug = self.add_spatial_jitter(sequence)
        aug = self.spatial_scale_and_rotate(aug)
        aug = self.apply_frame_dropout(aug)
        aug = self.temporal_speed_stretch(aug)
        return aug

    def augment_dataset(self, X: np.ndarray, y: np.ndarray, samples_per_sequence: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augments batch of keypoint sequences of shape (N, T, V, C).
        """
        N, T, V, C = X.shape
        logger.info(f"Augmenting {N} keypoint sequences ({samples_per_sequence}x expansion)...")

        aug_X_list = [X]
        aug_y_list = [y]

        for _ in range(samples_per_sequence - 1):
            new_X = np.zeros_like(X)
            for i in range(N):
                new_X[i] = self.augment_sequence(X[i])
            aug_X_list.append(new_X)
            aug_y_list.append(y.copy())

        X_augmented = np.concatenate(aug_X_list, axis=0)
        y_augmented = np.concatenate(aug_y_list, axis=0)

        logger.info(f"Sequence augmentation complete: New shape X = {X_augmented.shape}, y = {y_augmented.shape}")
        return X_augmented, y_augmented
