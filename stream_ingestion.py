#!/usr/bin/env python3
"""
CLI Tool: Threaded / Asynchronous RTSP Ingestion Engine (`stream_ingestion.py`)
Usage:
    python stream_ingestion.py --source 0 --fps 12 --buffer 5
"""

import argparse
from pathlib import Path
import sys
import time
import cv2

from src.pipeline.stream_ingestion import RTSPStreamIngestor
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("StreamIngestionCLI")


def main():
    parser = argparse.ArgumentParser(description="Project Vigilance - Threaded RTSP Stream Ingestor")
    parser.add_argument("--config", type=str, default="configs/pipeline_config.yaml", help="Path to pipeline config YAML")
    parser.add_argument("--source", type=str, default="0", help="Camera URL/index or video file path")
    parser.add_argument("--fps", type=float, default=12.0, help="Target sampling FPS")
    parser.add_argument("--buffer", type=int, default=5, help="Max circular frame buffer capacity")

    args = parser.parse_args()

    ingestor = RTSPStreamIngestor(
        stream_source=args.source,
        stream_name="CLI_Stream",
        target_fps=args.fps,
        buffer_max_size=args.buffer
    )

    logger.info("Starting RTSP Stream Ingestor test... Press Ctrl+C to exit.")
    ingestor.start()

    try:
        start_t = time.time()
        frames_fetched = 0
        while time.time() - start_t < 10.0:  # Run test for 10 seconds
            data = ingestor.get_latest_frame()
            if data is not None:
                frame_idx, ts, frame = data
                frames_fetched += 1
                if frames_fetched % 15 == 0:
                    logger.info(
                        f"Fetched Frame {frame_idx} | Shape: {frame.shape} | "
                        f"Ingestion FPS: {ingestor.ingestion_fps:.1f} | Dropped: {ingestor.total_frames_dropped}"
                    )
            time.sleep(0.02)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        ingestor.stop()

    logger.info(
        f"Test Complete: Read {ingestor.total_frames_read} frames | "
        f"Sampled {ingestor.total_frames_sampled} | Dropped {ingestor.total_frames_dropped}"
    )


if __name__ == "__main__":
    main()
