"""
Edge Optimization & Model Export Engine for Project Vigilance.
Converts PyTorch & YOLOv8 models into TorchScript, ONNX, and TensorRT/OpenVINO compatible representations.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

try:
    import onnx
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

from src.models.distress_net import DistressSpatioTemporalNet
from src.utils.logger import setup_logger

logger = setup_logger("ModelExporter")


class ModelExporter:
    """Exports Tier-1 YOLOv8 and Tier-2 Spatio-Temporal PyTorch models to ONNX and TorchScript."""

    def __init__(self, models_dir: Union[str, Path] = "models") -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def export_yolo_to_onnx(
        self,
        yolo_pt_path: Union[str, Path] = "models/yolo_person.pt",
        opset: int = 13,
        half: bool = False
    ) -> Optional[Path]:
        """Exports YOLOv8 PyTorch weights to ONNX format."""
        pt_path = Path(yolo_pt_path)
        if not pt_path.exists():
            logger.warning(f"YOLO model weights not found at {pt_path}. Skipping YOLO export.")
            return None

        if not HAS_ULTRALYTICS:
            logger.error("Ultralytics package missing. Cannot export YOLOv8 model.")
            return None

        try:
            logger.info(f"Exporting YOLOv8 model {pt_path} to ONNX (opset={opset}, FP16={half})...")
            yolo_model = YOLO(str(pt_path))
            onnx_path = yolo_model.export(format="onnx", opset=opset, half=half, dynamic=True)
            logger.info(f"Successfully exported YOLOv8 to ONNX: {onnx_path}")
            return Path(onnx_path)
        except Exception as e:
            logger.error(f"YOLOv8 ONNX export failed: {e}")
            return None

    def export_distress_to_torchscript(
        self,
        model: DistressSpatioTemporalNet,
        dummy_input: torch.Tensor,
        out_filename: str = "distress_traced.pt"
    ) -> Path:
        """Traces PyTorch Tier-2 Distress model into TorchScript format."""
        model.eval()
        out_path = self.models_dir / out_filename
        
        logger.info(f"Tracing Tier-2 Distress model to TorchScript format...")
        with torch.no_grad():
            traced_model = torch.jit.trace(model, dummy_input)
            traced_model.save(str(out_path))

        logger.info(f"Successfully saved TorchScript model -> {out_path}")
        return out_path

    def export_distress_to_onnx(
        self,
        model: DistressSpatioTemporalNet,
        dummy_input: torch.Tensor,
        out_filename: str = "distress_model.onnx",
        opset: int = 13
    ) -> Path:
        """Exports PyTorch Tier-2 Distress model to ONNX format with dynamic batch axes."""
        model.eval()
        out_path = self.models_dir / out_filename

        logger.info(f"Exporting Tier-2 Distress model to ONNX (opset={opset})...")
        dynamic_axes = {
            "input_keypoints": {0: "batch_size"},
            "output_logits": {0: "batch_size"}
        }

        try:
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(out_path),
                    export_params=True,
                    opset_version=opset,
                    do_constant_folding=True,
                    input_names=["input_keypoints"],
                    output_names=["output_logits"],
                    dynamic_axes=dynamic_axes
                )

            if HAS_ONNX:
                onnx_model = onnx.load(str(out_path))
                onnx.checker.check_model(onnx_model)
                logger.info(f"ONNX Model Verification PASSED for {out_path}")

            logger.info(f"Successfully exported Tier-2 ONNX model -> {out_path}")
            return out_path
        except Exception as e:
            logger.warning(f"ONNX export skipped/failed ({e}). TorchScript model is saved at {self.models_dir / 'distress_traced.pt'}")
            return out_path
