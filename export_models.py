#!/usr/bin/env python3
"""
CLI Tool: Edge Optimization & Model Export Engine (`export_models.py`)
Usage:
    python export_models.py --models_dir models --opset 13
"""

import argparse
from pathlib import Path
import sys
import torch

from src.export.exporter import ModelExporter
from src.models.distress_net import DistressSpatioTemporalNet
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("ExportModelsCLI")


def main():
    parser = argparse.ArgumentParser(description="Project Vigilance - Edge Model Exporter")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory where models are saved")
    parser.add_argument("--yolo_ckpt", type=str, default="models/yolo_person.pt", help="Path to trained YOLO PyTorch weights")
    parser.add_argument("--distress_ckpt", type=str, default="models/distress_model.pt", help="Path to trained Distress PyTorch weights")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")

    args = parser.parse_args()

    exporter = ModelExporter(models_dir=args.models_dir)

    # 1. Export Tier-1 YOLOv8 to ONNX
    yolo_ckpt_path = Path(args.yolo_ckpt)
    if yolo_ckpt_path.exists():
        exporter.export_yolo_to_onnx(yolo_pt_path=yolo_ckpt_path, opset=args.opset)
    else:
        logger.warning(f"Tier-1 YOLO checkpoint not found at {yolo_ckpt_path}. Skipping YOLO export.")

    # 2. Instantiate and export Tier-2 Distress Model
    distress_model = DistressSpatioTemporalNet(num_keypoints=33, channels=3, num_classes=4)
    distress_ckpt_path = Path(args.distress_ckpt)
    if distress_ckpt_path.exists():
        checkpoint = torch.load(distress_ckpt_path, map_location="cpu", weights_only=False)
        distress_model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded trained Tier-2 weights from {distress_ckpt_path}")
    else:
        logger.warning("No trained Tier-2 weights found. Exporting randomly initialized model structure...")

    dummy_seq = torch.randn(1, 30, 33, 3)

    # Export to TorchScript
    exporter.export_distress_to_torchscript(distress_model, dummy_seq, out_filename="distress_traced.pt")

    # Export to ONNX
    exporter.export_distress_to_onnx(distress_model, dummy_seq, out_filename="distress_model.onnx", opset=args.opset)

    logger.info("=== EDGE MODEL EXPORT COMPLETE ===")


if __name__ == "__main__":
    main()
