#!/usr/bin/env python3
"""
Project Vigilance - Stage 3 Real-Time Stream Ingestion & Multi-Stage Pipeline Integration
Demonstrates:
1. Threaded RTSP Ingestion with stale frame purging
2. Multi-Object Tracking & rolling keypoint sequence FIFO buffers (30 frames)
3. Two-Tier Chaining Inference (YOLOv8 + In-House Distress Model)
4. Telemetry HUD Rendering (Ingestion FPS, Inference FPS, Latency ms)
"""

from pathlib import Path
import time
import cv2

from src.pipeline.pipeline_orchestrator import TwoTierPipelineOrchestrator
from src.pipeline.pipeline_viewer import PipelineTelemetryViewer
from src.pipeline.stream_ingestion import RTSPStreamIngestor
from src.utils.logger import setup_logger

logger = setup_logger("Stage3DemoPipeline")


def main():
    base_dir = Path(__file__).parent.resolve()
    logger.info("Initializing Stage 3 Pipeline Demonstration...")

    video_path = base_dir / "data" / "raw_videos" / "demo_cctv_feed.mp4"
    if not video_path.exists():
        logger.warning("Demo video path not found. Initializing fallback ingestion source '0'...")
        video_source = "0"
    else:
        video_source = str(video_path)

    # 1. Instantiate Ingestor
    ingestor = RTSPStreamIngestor(
        stream_source=video_source,
        stream_name="Demo_CCTV_Feed",
        target_fps=12.0,
        buffer_max_size=5
    )

    # 2. Instantiate Two-Tier Pipeline Orchestrator
    orchestrator = TwoTierPipelineOrchestrator(
        yolo_model_path=base_dir / "models" / "yolo_person.pt",
        distress_model_path=base_dir / "models" / "distress_model.pt",
        detection_confidence=0.45,
        distress_confidence_threshold=0.65
    )

    # 3. Instantiate Visual Telemetry Viewer
    viewer = PipelineTelemetryViewer()

    logger.info("Launching Stage 3 Integration Pipeline...")
    ingestor.start()

    start_time = time.time()
    frames_processed = 0

    try:
        while time.time() - start_time < 15.0:  # Run demonstration for 15 seconds
            data = ingestor.get_latest_frame()
            if data is not None:
                frame_idx, ts, frame = data
                frames_processed += 1

                # Execute two-tier inference chaining
                pipeline_out = orchestrator.process_frame(frame, timestamp=ts)

                # Render visual telemetry HUD
                rendered_frame = viewer.render_frame(frame, pipeline_out, ingestion_fps=ingestor.ingestion_fps)

                if frames_processed % 10 == 0:
                    logger.info(
                        f"Processed Frame #{frame_idx:04d} | Ingestion: {ingestor.ingestion_fps:.1f} FPS | "
                        f"Pipeline Latency: {pipeline_out['latency_ms']:.2f} ms ({pipeline_out['fps']:.1f} FPS) | "
                        f"Active Tracks: {len(pipeline_out['tracked_persons'])} | Alerts: {len(pipeline_out['alerts'])}"
                    )
            else:
                time.sleep(0.01)

            if not ingestor.running and not ingestor.frame_buffer:
                break
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user.")
    finally:
        ingestor.stop()
        viewer.close()

    logger.info("=== STAGE 3 PIPELINE DEMONSTRATION COMPLETE ===")
    logger.info(f"Total Frames Processed: {frames_processed}")


if __name__ == "__main__":
    main()
