"""
Threaded / Asynchronous RTSP Stream Ingestion Engine for Project Vigilance.
Handles low-latency RTSP/file video ingestion, non-blocking frame buffer management,
automatic reconnection with exponential backoff, and dynamic FPS sampling.
"""

from collections import deque
from pathlib import Path
import threading
import time
from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger("RTSPStreamIngestor")


class RTSPStreamIngestor:
    """Threaded RTSP / Video Feed Ingestor with stale frame purging & auto-reconnect."""

    def __init__(
        self,
        stream_source: Union[str, int],
        stream_name: str = "CCTV_Stream",
        target_fps: float = 12.0,
        buffer_max_size: int = 5,
        reconnect_attempts: int = 5,
        reconnect_delay_sec: float = 2.0
    ) -> None:
        """
        Args:
            stream_source: Path to video file, integer device index, or RTSP URL.
            stream_name: Friendly identifier for logging.
            target_fps: Target sampling rate for downstream inference.
            buffer_max_size: Size of circular frame buffer. Older frames dropped if full.
            reconnect_attempts: Max consecutive connection retries.
            reconnect_delay_sec: Base delay in seconds between retries.
        """
        # Parse numeric camera indices if passed as string digits e.g. "0"
        if isinstance(stream_source, str) and stream_source.isdigit():
            self.stream_source = int(stream_source)
        else:
            self.stream_source = stream_source

        self.stream_name = stream_name
        self.target_fps = target_fps
        self.buffer_max_size = buffer_max_size
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay_sec = reconnect_delay_sec

        # Circular frame buffer (thread-safe deque)
        self.frame_buffer = deque(maxlen=buffer_max_size)
        self.lock = threading.Lock()

        # Threading state variables
        self.running = False
        self.connected = False
        self.thread: Optional[threading.Thread] = None

        # Statistics
        self.cap_fps = 30.0
        self.total_frames_read = 0
        self.total_frames_sampled = 0
        self.total_frames_dropped = 0
        self.ingestion_fps = 0.0

    def _connect() -> Optional[cv2.VideoCapture]:
        """Attempts connection to video source with exponential backoff."""
        attempt = 0
        while attempt < self.reconnect_attempts:
            attempt += 1
            logger.info(f"[{self.stream_name}] Connecting to source (Attempt {attempt}/{self.reconnect_attempts})...")
            
            cap = cv2.VideoCapture(self.stream_source)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                self.cap_fps = fps if (fps > 0 and not np.isnan(fps)) else 30.0
                self.connected = True
                logger.info(f"[{self.stream_name}] Connected successfully! (Source FPS: {self.cap_fps:.1f})")
                return cap

            logger.warning(f"[{self.stream_name}] Connection failed. Retrying in {self.reconnect_delay_sec}s...")
            time.sleep(self.reconnect_delay_sec * (1.5 ** (attempt - 1)))

        logger.error(f"[{self.stream_name}] Failed to connect after {self.reconnect_attempts} attempts.")
        self.connected = False
        return None

    def _reader_loop() -> None:
        """Background worker thread continuously capturing frames."""
        cap = self._connect()
        if cap is None:
            self.running = False
            return

        sample_interval = max(1, int(round(self.cap_fps / self.target_fps)))
        frame_idx = 0
        fps_timer = time.time()
        frames_in_second = 0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"[{self.stream_name}] Frame read failed or EOF reached. Reconnecting...")
                cap.release()
                self.connected = False
                cap = self._connect()
                if cap is None:
                    break
                continue

            frame_idx += 1
            self.total_frames_read += 1
            frames_in_second += 1

            # Update FPS metric
            if time.time() - fps_timer >= 1.0:
                self.ingestion_fps = frames_in_second / (time.time() - fps_timer)
                frames_in_second = 0
                fps_timer = time.time()

            # Dynamic sampling filter
            if frame_idx % sample_interval != 0:
                continue

            timestamp = time.time()
            with self.lock:
                if len(self.frame_buffer) == self.buffer_max_size:
                    self.total_frames_dropped += 1  # Purge oldest stale frame
                self.frame_buffer.append((frame_idx, timestamp, frame))
                self.total_frames_sampled += 1

        if cap is not None:
            cap.release()
        self.connected = False
        logger.info(f"[{self.stream_name}] Ingestion thread terminated.")

    def start(self) -> None:
        """Starts background frame reader thread."""
        if self.running:
            logger.warning(f"[{self.stream_name}] Ingestor already running.")
            return

        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()
        logger.info(f"[{self.stream_name}] Ingestion thread launched.")

    def get_latest_frame(self) -> Optional[Tuple[int, float, np.ndarray]]:
        """
        Non-blocking call returning the latest sampled frame from buffer.
        Returns Tuple of (frame_idx, timestamp, frame_bgr) or None if buffer is empty.
        """
        with self.lock:
            if not self.frame_buffer:
                return None
            return self.frame_buffer.popleft()  # FIFO consumption

    def stop(self) -> None:
        """Gracefully stops ingestion thread."""
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        logger.info(f"[{self.stream_name}] Ingestor stopped cleanly.")
