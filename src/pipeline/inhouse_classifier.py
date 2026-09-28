"""
Non-Destructive Inference Wrapper for In-House Spatio-Temporal Distress Model.
Provides translation & scale-invariant coordinate normalization and supports both PyTorch and ONNX Runtime.
"""

from collections import deque
from pathlib import Path
from typing import Deque, Dict, Tuple, Union
import numpy as np
import torch

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

from models.inhouse_distress_model import InHouseDistressNet
from src.utils.logger import setup_logger

logger = setup_logger("InHouseClassifier")


class InHouseClassifier:
    """Non-destructive inference engine for In-House Spatio-Temporal Distress Model."""

    LABEL_MAP = {
        0: "Normal",
        1: "Universal Help Signal",
        2: "Aggression/Struggle",
        3: "Fall"
    }

    def __init__(
        self,
        model_path: Union[str, Path] = "models/inhouse_distress.onnx",
        sequence_length: int = 30,
        num_keypoints: int = 17,
        channels: int = 3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:
        """
        Args:
            model_path: Path to .onnx or .pt weights file.
            sequence_length: Temporal window length (30 frames).
            num_keypoints: Number of skeletal joints (17 or 33).
            channels: Channels per keypoint (3 for [x, y, confidence]).
            device: Computing device ("cuda" or "cpu").
        """
        self.path = Path(model_path)
        self.sequence_length = sequence_length
        self.num_keypoints = num_keypoints
        self.channels = channels
        self.device = device

        self.ort_session = None
        self.pt_model = None

        self._load_engine()

    def _load_engine(self) -> None:
        """Loads ONNX Runtime session if .onnx file exists, else loads PyTorch module."""
        if self.path.suffix == ".onnx" and HAS_ONNX and self.path.exists():
            logger.info(f"Loading InHouseClassifier via ONNX Runtime engine: {self.path}")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
            self.ort_session = ort.InferenceSession(str(self.path), providers=providers)
        else:
            pt_path = self.path if self.path.suffix == ".pt" else self.path.parent / "inhouse_distress.pt"
            logger.info(f"Loading InHouseClassifier via PyTorch engine: {pt_path}")
            self.pt_model = InHouseDistressNet.from_pretrained(pt_path, device=self.device)

    @staticmethod
    def normalize_keypoint_sequence(sequence: np.ndarray) -> np.ndarray:
        """
        Normalizes skeletal keypoint coordinates relative to torso/root center (hip joint).
        Makes prediction distance & camera translation invariant.

        Input sequence shape: (30, V, C) where C >= 2
        Output sequence shape: (30, V, C)
        """
        norm_seq = sequence.copy()
        T, V, C = norm_seq.shape

        # Select root joint index (Hip center: index 11 for COCO or 23 for MediaPipe)
        if V == 17:
            root_idx = 11  # Left hip / root landmark in 17-keypoint COCO
        else:
            root_idx = 23  # Left hip landmark in 33-keypoint MediaPipe

        for t in range(T):
            root_x, root_y = norm_seq[t, root_idx, 0], norm_seq[t, root_idx, 1]
            if root_x > 0 and root_y > 0:
                norm_seq[t, :, 0] -= root_x
                norm_seq[t, :, 1] -= root_y

        return norm_seq

    def predict(
        self,
        keypoint_buffer: Union[np.ndarray, Deque[np.ndarray]]
    ) -> Tuple[str, float]:
        """
        Executes forward inference pass on keypoint sliding-window buffer.

        Args:
            keypoint_buffer: Array or deque of shape (30, V, C) or (30, V*C)

        Returns:
            Tuple of (predicted_label: str, confidence_score: float)
        """
        if isinstance(keypoint_buffer, deque):
            seq = np.array(keypoint_buffer, dtype=np.float32)
        else:
            seq = np.array(keypoint_buffer, dtype=np.float32)

        if seq.shape[0] < self.sequence_length:
            return "Normal", 1.0

        # Adapt 33 keypoints -> 17 keypoints if model expects 17
        if seq.shape[1] == 33 and self.num_keypoints == 17:
            coco_idx = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
            seq = seq[:, coco_idx, :]

        # Distance-invariant normalization
        norm_seq = self.normalize_keypoint_sequence(seq)

        # Batch dimension expansion: (1, 30, V, C)
        inp_batch = np.expand_dims(norm_seq, axis=0).astype(np.float32)

        if self.ort_session is not None:
            input_name = self.ort_session.get_inputs()[0].name
            logits = self.ort_session.run(None, {input_name: inp_batch})[0]
            exp_logits = np.exp(logits - np.max(logits))
            probs = (exp_logits / np.sum(exp_logits, axis=-1, keepdims=True))[0]
        else:
            with torch.no_grad():
                inp_torch = torch.from_numpy(inp_batch).to(self.device)
                logits_torch = self.pt_model(inp_torch)
                probs = torch.softmax(logits_torch, dim=1).cpu().numpy()[0]

        pred_class_id = int(np.argmax(probs))
        confidence = float(probs[pred_class_id])
        label_str = self.LABEL_MAP.get(pred_class_id, "Normal")

        return label_str, confidence
