#!/usr/bin/env python3
"""
CLI Tool: Two-Tier Inference Chaining Pipeline Orchestrator (`pipeline_orchestrator.py`)
Usage:
    python pipeline_orchestrator.py --config configs/pipeline_config.yaml
"""

import argparse
from pathlib import Path
import cv2
import numpy as np

from src.pipeline.pipeline_orchestrator import TwoTierPipelineOrchestrator
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("PipelineOrchestratorCLI")


def main():
    parser = argparse.ArgumentParser(description="Project Vigilance - Two-Tier Pipeline Orchestrator")
    parser.add_argument("--config", type=str, default="configs/pipeline_config.yaml", help="Path to config YAML")
    parser.add_argument("--video", type=str, default="data/raw_videos/demo_cctv_feed.mp4", help="Video file or stream source")
    
    args = parser.parse_args()

    orchestrator = TwoTierPipelineOrchestrator()

    video_path = Path(args.video)
    if not video_path.exists():
        logger.warning(f"Video '{video_path}' not found. Generating synthetic test frame sequence...")
        dummy_frame = np.full((720, 1280, 3), 40, dtype=np.uint8)
        cv2.rectangle(dummy_frame, (350, 200), (450, 450), (0, 255, 120), -1)
        res = orchestrator.process_frame(dummy_frame)
        logger.info(f"Synthetic Frame Processing Result: {res}")
        return

    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0

    logger.info(f"Processing video feed: {video_path}...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        res = orchestrator.process_frame(frame, timestamp=cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)

        if frame_count % 15 == 0:
            logger.info(
                f"Frame {frame_count:04d} | Active Trackers: {len(res['tracked_persons'])} | "
                f"Alerts: {len(res['alerts'])} | Latency: {res['latency_ms']:.2f} ms ({res['fps']:.1f} FPS)"
            )

    cap.release()
    logger.info("Pipeline processing finished.")


if __name__ == "__main__":
    main()
