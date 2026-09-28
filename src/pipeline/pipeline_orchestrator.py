"""
Two-Tier Inference Chaining Pipeline Orchestrator for Project Vigilance.
Chains Tier-1 YOLOv8 Person Detection -> Multi-Object Tracking & Keypoint Extraction -> Tier-2 Distress Classification.
"""

from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch

try:
    import onnxruntime as ort
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

from src.models.distress_net import DistressSpatioTemporalNet
from src.pipeline.tracker_manager import TrackletManager
from src.utils.logger import setup_logger

logger = setup_logger("PipelineOrchestrator")


class TwoTierPipelineOrchestrator:
    """Orchestrates real-time chaining of Tier-1 YOLO Person Localizer and Tier-2 Spatio-Temporal Distress Classifier."""

    CLASS_MAP = {
        0: "Normal",
        1: "Universal Help Signal",
        2: "Aggression/Struggle",
        3: "Fall"
    }

    def __init__(
        self,
        yolo_model_path: Union[str, Path] = "models/yolo_person.pt",
        distress_model_path: Union[str, Path] = "models/distress_model.pt",
        detection_confidence: float = 0.45,
        distress_confidence_threshold: float = 0.65,
        window_length: int = 30,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:
        """
        Args:
            yolo_model_path: Path to fine-tuned Tier-1 YOLO model weights (.pt or .onnx).
            distress_model_path: Path to trained Tier-2 distress model weights (.pt or .onnx).
            detection_confidence: Minimum confidence threshold for YOLO person detection.
            distress_confidence_threshold: Minimum confidence threshold to trigger distress alert.
            window_length: Temporal sequence window length.
            device: Computing device ("cuda" or "cpu").
        """
        self.detection_confidence = detection_confidence
        self.distress_confidence_threshold = distress_confidence_threshold
        self.window_length = window_length
        self.device = device

        # Initialize Tracklet & Keypoint Manager
        self.tracker_manager = TrackletManager(window_length=window_length)

        # 1. Load Tier-1 YOLO Model
        self.yolo_model = self._load_yolo_model(yolo_model_path)

        # 2. Load Tier-2 Distress Model
        self.distress_model, self.distress_onnx_session = self._load_distress_model(distress_model_path)

        # Performance Metrics
        self.inference_fps = 0.0
        self.total_pipeline_latency_ms = 0.0

    def _load_yolo_model(self, model_path: Union[str, Path]) -> Any:
        path = Path(model_path)
        if not path.exists():
            fallback_pt = path.parent / "yolo_person.pt"
            if fallback_pt.exists():
                path = fallback_pt
            else:
                logger.warning(f"YOLO model weights not found at {path}. Using pretrained base 'yolov8n.pt'")
                path = Path("yolov8n.pt")

        if HAS_ULTRALYTICS:
            logger.info(f"Loading Tier-1 YOLOv8 model from {path}...")
            return YOLO(str(path))
        else:
            logger.error("Ultralytics package is missing.")
            return None

    def _load_distress_model(
        self,
        model_path: Union[str, Path]
    ) -> Tuple[Optional[DistressSpatioTemporalNet], Optional[Any]]:
        path = Path(model_path)
        
        # Check for ONNX runtime model first
        onnx_path = path.parent / "distress_model.onnx"
        if HAS_ONNXRUNTIME and onnx_path.exists():
            logger.info(f"Loading Tier-2 Distress model via ONNX Runtime: {onnx_path}")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
            session = ort.InferenceSession(str(onnx_path), providers=providers)
            return None, session

        # Fall back to PyTorch model
        pt_path = path if path.suffix == ".pt" else path.parent / "distress_model.pt"
        logger.info(f"Loading Tier-2 Distress PyTorch model from {pt_path}...")
        model = DistressSpatioTemporalNet(num_keypoints=33, channels=3, num_classes=4)
        if pt_path.exists():
            try:
                ckpt = torch.load(pt_path, map_location=self.device, weights_only=False)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"])
                else:
                    model.load_state_dict(ckpt)
                logger.info("Successfully loaded Tier-2 PyTorch weights.")
            except Exception as e:
                logger.warning(f"Could not load PyTorch checkpoint: {e}. Using initialized architecture.")

        model.to(self.device)
        model.eval()
        return model, None

    def predict_distress_sequence(self, sequence_tensor: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Runs Tier-2 inference on keypoint sequence tensor of shape (30, 33, 3).
        Returns Tuple of (predicted_class_id, max_confidence_score, class_probabilities_array).
        """
        # Batch dimension expansion: (1, 30, 33, 3)
        inp_batch = np.expand_dims(sequence_tensor, axis=0).astype(np.float32)

        if self.distress_onnx_session is not None:
            input_name = self.distress_onnx_session.get_inputs()[0].name
            logits = self.distress_onnx_session.run(None, {input_name: inp_batch})[0]
            # Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = (exp_logits / np.sum(exp_logits, axis=-1, keepdims=True))[0]
        else:
            with torch.no_grad():
                inp_torch = torch.from_numpy(inp_batch).to(self.device)
                logits_torch = self.distress_model(inp_torch)
                probs = torch.softmax(logits_torch, dim=1).cpu().numpy()[0]

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        return pred_class, confidence, probs

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp: float = 0.0
    ) -> Dict[str, Any]:
        """
        Executes end-to-end two-tier pipeline on input video frame.

        Returns dictionary containing:
        - tracked_persons: List of dicts {track_id, bbox, gesture_class, confidence, alert_flag}
        - alerts: List of high-priority distress alert events
        - latency_ms: Execution latency in milliseconds
        """
        start_time = time.perf_counter()
        H, W = frame_bgr.shape[:2]

        # STEP A: Tier-1 YOLOv8 Person Detection
        detections = []
        scores = []
        if self.yolo_model is not None:
            results = self.yolo_model.predict(
                frame_bgr,
                conf=self.detection_confidence,
                classes=[0],  # Class 0: Person
                verbose=False
            )
            if results and len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].cpu().numpy())
                    detections.append(xyxy)
                    scores.append(conf)

        # STEP B & C: Multi-Object Tracking & Keypoint Sequence Buffer Update
        active_tracks = self.tracker_manager.update_frame(frame_bgr, detections, scores)

        tracked_persons = []
        alert_events = []

        for track_id, (bbox, seq_array) in active_tracks.items():
            gesture_class_id = 0  # Default Normal
            confidence = 1.0
            alert_flag = False

            # STEP D: Run Tier-2 Distress Classifier if sequence buffer full (30 frames)
            if seq_array is not None:
                gesture_class_id, confidence, _ = self.predict_distress_sequence(seq_array)
                
                # Check alert condition
                if gesture_class_id > 0 and confidence >= self.distress_confidence_threshold:
                    alert_flag = True
                    alert_events.append({
                        "timestamp": timestamp or time.time(),
                        "track_id": track_id,
                        "gesture_id": gesture_class_id,
                        "gesture_name": self.CLASS_MAP.get(gesture_class_id, "Unknown"),
                        "confidence": confidence,
                        "bbox": bbox
                    })

            tracked_persons.append({
                "track_id": track_id,
                "bbox": bbox,
                "gesture_id": gesture_class_id,
                "gesture_name": self.CLASS_MAP.get(gesture_class_id, "Normal"),
                "confidence": confidence,
                "is_alert": alert_flag
            })

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0
        self.total_pipeline_latency_ms = latency_ms
        self.inference_fps = 1000.0 / max(1e-5, latency_ms)

        return {
            "timestamp": timestamp,
            "tracked_persons": tracked_persons,
            "alerts": alert_events,
            "latency_ms": latency_ms,
            "fps": self.inference_fps
        }
