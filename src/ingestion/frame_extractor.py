"""
Frame Extraction & CCTV Ingestion Tool for Project Vigilance.
Ingests video streams/files, performs dynamic frame sampling, and filters duplicate frames via SSIM/diff.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from tqdm import tqdm

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

from src.utils.logger import setup_logger

logger = setup_logger("FrameExtractor")


class FrameExtractor:
    """Ingests raw CCTV video feeds and extracts non-redundant keyframes."""

    def __init__(
        self,
        target_fps: float = 12.0,
        similarity_method: str = "ssim",
        similarity_threshold: float = 0.94,
        resize_dim: Optional[Tuple[int, int]] = (1280, 720),
        save_format: str = "png"
    ) -> None:
        """
        Args:
            target_fps: Target sampling rate in frames per second.
            similarity_method: Frame similarity metric ("ssim" or "diff").
            similarity_threshold: Threshold above which frame is discarded as duplicate.
            resize_dim: Optional (width, height) to resize extracted frames.
            save_format: Output image format ("png" or "jpg").
        """
        self.target_fps = target_fps
        self.similarity_method = similarity_method.lower()
        self.similarity_threshold = similarity_threshold
        self.resize_dim = resize_dim
        self.save_format = save_format

        if self.similarity_method == "ssim" and not HAS_SKIMAGE:
            logger.warning("skimage not installed. Falling back to background difference metric.")
            self.similarity_method = "diff"

    def compute_similarity(self, frame_a: np.ndarray, frame_b: np.ndarray) -> float:
        """
        Computes similarity score between two grayscale frames.
        Returns value in [0, 1] where 1.0 indicates identical images.
        """
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

        # Downsample for faster metric calculation if large
        if gray_a.shape[0] > 360 or gray_a.shape[1] > 640:
            gray_a = cv2.resize(gray_a, (320, 180))
            gray_b = cv2.resize(gray_b, (320, 180))

        if self.similarity_method == "ssim":
            score, _ = ssim(gray_a, gray_b, full=True)
            return float(score)
        else:
            # Normalized absolute difference score inverted to [0, 1] similarity
            diff = cv2.absdiff(gray_a, gray_b)
            mean_diff = np.mean(diff) / 255.0
            similarity = 1.0 - mean_diff
            return float(similarity)

    def process_video(
        self,
        video_source: Union[str, Path, int],
        output_dir: Union[str, Path],
        video_prefix: Optional[str] = None
    ) -> Dict[str, Union[int, float]]:
        """
        Reads input video file or RTSP stream and extracts sampled non-duplicate frames.

        Args:
            video_source: Path to video file or RTSP stream URL / device index.
            output_dir: Directory where extracted frames will be stored.
            video_prefix: Optional custom prefix for extracted filenames.

        Returns:
            Dictionary containing extraction metadata and statistics.
        """
        source_str = str(video_source)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(source_str)
        if not cap.isOpened():
            logger.error(f"Failed to open video source: {source_str}")
            raise ValueError(f"Cannot open video source: {source_str}")

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps <= 0 or np.isnan(original_fps):
            original_fps = 30.0  # Default fallback for unmetered RTSP streams

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 100000  # Stream limit cap

        # Calculate sampling step interval
        sample_step = max(1, int(round(original_fps / self.target_fps)))
        logger.info(
            f"Processing: {source_str} | Source FPS: {original_fps:.1f} | "
            f"Target FPS: {self.target_fps:.1f} | Frame Step: {sample_step}"
        )

        prefix = video_prefix or (Path(source_str).stem if Path(source_str).is_file() else "stream")
        
        frame_idx = 0
        saved_count = 0
        skipped_fps_count = 0
        skipped_duplicate_count = 0
        last_saved_frame: Optional[np.ndarray] = None

        pbar = tqdm(total=total_frames if total_frames < 99999 else None, desc=f"Extracting {prefix}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                pbar.update(1)

                # 1. Dynamic FPS sampling check
                if frame_idx % sample_step != 0:
                    skipped_fps_count += 1
                    continue

                # 2. Resize if requested
                if self.resize_dim is not None:
                    curr_frame = cv2.resize(frame, self.resize_dim, interpolation=cv2.INTER_AREA)
                else:
                    curr_frame = frame.copy()

                # 3. Static/Duplicate Frame Filtering
                if last_saved_frame is not None:
                    similarity = self.compute_similarity(last_saved_frame, curr_frame)
                    if similarity >= self.similarity_threshold:
                        skipped_duplicate_count += 1
                        continue

                # 4. Save keyframe
                filename = f"{prefix}_frame_{frame_idx:06d}.{self.save_format}"
                out_file = output_path / filename
                cv2.imwrite(str(out_file), curr_frame)
                
                last_saved_frame = curr_frame
                saved_count += 1

        finally:
            cap.release()
            pbar.close()

        stats = {
            "total_frames_read": frame_idx,
            "saved_frames": saved_count,
            "skipped_fps_frames": skipped_fps_count,
            "skipped_duplicate_frames": skipped_duplicate_count,
            "reduction_ratio": float(1.0 - (saved_count / max(1, frame_idx)))
        }

        logger.info(
            f"Extraction Finished: Saved {saved_count}/{frame_idx} frames "
            f"({stats['reduction_ratio']*100:.1f}% reduction rate)."
        )
        return stats
