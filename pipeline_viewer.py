#!/usr/bin/env python3
"""
CLI Tool: Visual Debugger & Performance Telemetry (`pipeline_viewer.py`)
Usage:
    python pipeline_viewer.py --source data/raw_videos/demo_cctv_feed.mp4
"""

import argparse
from pathlib import Path
import time
import cv2
import numpy as np

from src.pipeline.pipeline_orchestrator import TwoTierPipelineOrchestrator
from src.pipeline.pipeline_viewer import PipelineTelemetryViewer
from src.pipeline.stream_ingestion import RTSPStreamIngestor
from src.utils.logger import setup_logger

logger = setup_logger("PipelineViewerCLI")


def main():
    parser = argparse.ArgumentParser(description="Project Vigilance - Visual Telemetry Viewer")
    parser.add_argument("--source", type=str, default="data/raw_videos/demo_cctv_feed.mp4", help="Video source or RTSP URL")
    parser.add_argument("--fps", type=float, default=12.0, help="Sampling rate FPS")

    args = parser.parse_args()

    ingestor = RTSPStreamIngestor(stream_source=args.source, target_fps=args.fps)
    orchestrator = TwoTierPipelineOrchestrator()
    viewer = PipelineTelemetryViewer()

    logger.info("Starting Visual Telemetry Pipeline... Press 'q' or ESC in window to exit.")
    ingestor.start()

    try:
        while True:
            data = ingestor.get_latest_frame()
            if data is not None:
                frame_idx, ts, frame = data
                pipeline_out = orchestrator.process_frame(frame, timestamp=ts)
                rendered_frame = viewer.render_frame(frame, pipeline_out, ingestion_fps=ingestor.ingestion_fps)
                
                keep_running = viewer.show(rendered_frame)
                if not keep_running:
                    break
            else:
                time.sleep(0.01)

            if not ingestor.running and ingestor.frame_buffer:
                break
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        ingestor.stop()
        viewer.close()

    logger.info("Telemetry viewer exited cleanly.")


if __name__ == "__main__":
    main()
