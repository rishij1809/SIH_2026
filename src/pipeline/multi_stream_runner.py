"""
Multi-Stream Support & Concurrency Manager for Project Vigilance.
Runs parallel worker threads/processes across multiple camera feeds with central alert aggregation.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any, Dict, List, Optional
import yaml

from src.pipeline.pipeline_orchestrator import TwoTierPipelineOrchestrator
from src.pipeline.stream_ingestion import RTSPStreamIngestor
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("MultiStreamRunner")


class CameraWorker(threading.Thread):
    """Worker thread running dedicated ingestion & inference pipeline for a single camera feed."""

    def __init__(
        self,
        camera_id: str,
        camera_cfg: Dict[str, Any],
        pipeline_cfg: Dict[str, Any],
        alert_callback: Optional[callable] = None
    ) -> None:
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.camera_name = camera_cfg.get("name", camera_id)
        self.url = camera_cfg.get("url", "0")
        self.target_fps = camera_cfg.get("target_fps", 12.0)
        self.alert_callback = alert_callback

        self.running = False
        self.ingestor = RTSPStreamIngestor(
            stream_source=self.url,
            stream_name=self.camera_name,
            target_fps=self.target_fps,
            buffer_max_size=camera_cfg.get("buffer_max_size", 5)
        )

        self.orchestrator = TwoTierPipelineOrchestrator(
            yolo_model_path=pipeline_cfg.get("yolo_model_path", "models/yolo_person.pt"),
            distress_model_path=pipeline_cfg.get("distress_model_path", "models/distress_model.pt"),
            detection_confidence=pipeline_cfg.get("detection_confidence", 0.45),
            distress_confidence_threshold=pipeline_cfg.get("distress_confidence_threshold", 0.65)
        )

        self.processed_frames = 0
        self.total_alerts = 0

    def run(self) -> None:
        """Main worker loop executing stream capture -> two-tier inference."""
        logger.info(f"[{self.camera_name}] Starting camera worker thread...")
        self.running = True
        self.ingestor.start()

        while self.running:
            data = self.ingestor.get_latest_frame()
            if data is not None:
                frame_idx, ts, frame = data
                pipeline_out = self.orchestrator.process_frame(frame, timestamp=ts)
                self.processed_frames += 1

                alerts = pipeline_out.get("alerts", [])
                if alerts:
                    self.total_alerts += len(alerts)
                    for alert in alerts:
                        logger.warning(
                            f"🚨 ALERT on [{self.camera_name}]! Track ID {alert['track_id']} -> "
                            f"Detected: {alert['gesture_name']} (Conf: {alert['confidence']*100:.1f}%)"
                        )
                        if self.alert_callback is not None:
                            self.alert_callback(self.camera_id, self.camera_name, alert)
            else:
                time.sleep(0.01)

        self.ingestor.stop()
        logger.info(f"[{self.camera_name}] Camera worker thread stopped cleanly.")

    def stop(self) -> None:
        self.running = False


class MultiStreamRunner:
    """Manages concurrent execution across multiple CCTV streams."""

    def __init__(self, config_path: str = "configs/pipeline_config.yaml") -> None:
        self.config_path = Path(config_path)
        self.cfg = load_config(self.config_path) if self.config_path.exists() else {}
        self.workers: Dict[str, CameraWorker] = {}
        self.running = False

    def on_alert_event(self, camera_id: str, camera_name: str, alert_data: Dict[str, Any]) -> None:
        """Central event dispatcher callback."""
        # Custom alert routing (logging, webhooks, MQTT, database storage)
        pass

    def start(self) -> None:
        """Launches dedicated workers for all configured cameras."""
        stream_cfgs = self.cfg.get("streams", {})
        pipeline_cfg = self.cfg.get("pipeline", {})

        if not stream_cfgs:
            logger.warning("No camera streams found in config! Registering default webcam stream...")
            stream_cfgs = {
                "cam_0": {
                    "name": "Default Camera 0",
                    "url": "0",
                    "target_fps": 12.0
                }
            }

        logger.info(f"Initializing MultiStreamRunner for {len(stream_cfgs)} camera feed(s)...")
        self.running = True

        for cam_id, cam_cfg in stream_cfgs.items():
            worker = CameraWorker(
                camera_id=cam_id,
                camera_cfg=cam_cfg,
                pipeline_cfg=pipeline_cfg,
                alert_callback=self.on_alert_event
            )
            self.workers[cam_id] = worker
            worker.start()

        # Signal handlers for graceful SIGINT/SIGTERM shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame) -> None:
        logger.info(f"Received shutdown signal ({signum}). Stopping all camera workers...")
        self.stop()
        sys.exit(0)

    def stop(self) -> None:
        """Stops all running camera worker threads."""
        self.running = False
        for cam_id, worker in self.workers.items():
            worker.stop()
        for cam_id, worker in self.workers.items():
            worker.join(timeout=3.0)
        logger.info("All camera stream workers stopped cleanly.")
