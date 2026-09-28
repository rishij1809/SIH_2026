#!/usr/bin/env python3
"""
CLI Tool: Multi-Object Tracking & Tracklet Buffer Management (`tracker_manager.py`)
Usage:
    python tracker_manager.py --window 30 --stale 30
"""

import argparse
from pathlib import Path
import numpy as np

from src.pipeline.tracker_manager import TrackletManager
from src.utils.logger import setup_logger

logger = setup_logger("TrackerManagerCLI")


def main():
    parser = argparse.ArgumentParser(description="Project Vigilance - Tracklet Manager Test")
    parser.add_argument("--window", type=int, default=30, help="Temporal window length")
    parser.add_argument("--stale", type=int, default=30, help="Max stale lost frames")

    args = parser.parse_args()

    manager = TrackletManager(window_length=args.window, max_stale_frames=args.stale)

    logger.info("Testing TrackletManager with simulated detections across 35 video frames...")

    dummy_frame = np.full((720, 1280, 3), 50, dtype=np.uint8)

    for frame_idx in range(1, 36):
        # Move synthetic bounding box across frame
        x1 = 200 + frame_idx * 5
        y1 = 150
        x2 = x1 + 100
        y2 = y1 + 250
        
        detections = [[x1, y1, x2, y2]]
        scores = [0.92]

        results = manager.update_frame(dummy_frame, detections, scores)

        for track_id, (bbox, seq) in results.items():
            ready_str = f"READY (Shape: {seq.shape})" if seq is not None else f"Filling... ({len(manager.keypoint_buffers[track_id])}/{args.window})"
            if frame_idx % 5 == 0 or seq is not None:
                logger.info(f"Frame {frame_idx:02d} | Track ID {track_id} | BBox: {bbox} | Sequence Buffer: {ready_str}")

    logger.info("TrackletManager test completed successfully!")


if __name__ == "__main__":
    main()
