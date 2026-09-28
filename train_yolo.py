#!/usr/bin/env python3
"""
Tier 1: YOLOv8 Spatial Localization Fine-Tuning Script (`train_yolo.py`).
Fine-tunes lightweight YOLOv8 (yolov8n/yolov8s) for person detection on CCTV feeds.
Saves model checkpoints, validation metrics, confusion matrices, and PR curves.
"""

import argparse
from pathlib import Path
import sys
import yaml

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("TrainYOLO")


def main():
    parser = argparse.ArgumentParser(description="Project Vigilance - Tier 1 YOLOv8 Person Detector Fine-Tuning")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to pipeline config YAML")
    parser.add_argument("--data", type=str, default="configs/data.yaml", help="Path to YOLO data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model weights (yolov8n.pt / yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--project", type=str, default="models", help="Output project directory")
    parser.add_argument("--name", type=str, default="yolo_person", help="Run name")

    args = parser.parse_args()

    if not HAS_ULTRALYTICS:
        logger.error("Ultralytics package is not installed. Please run 'pip install ultralytics'.")
        sys.exit(1)

    cfg = load_config(args.config) if Path(args.config).exists() else {}
    data_yaml_path = Path(args.data)

    if not data_yaml_path.exists():
        logger.error(f"YOLO dataset config '{data_yaml_path}' not found! Run 'prepare_yolo.py' first.")
        sys.exit(1)

    logger.info(f"Loading base YOLOv8 model: {args.model}")
    model = YOLO(args.model)

    # Execute Ultralytics training with edge-optimized hyperparameters
    logger.info(f"Starting Tier-1 YOLOv8 fine-tuning for {args.epochs} epochs...")
    results = model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3.0,
        weight_decay=0.0005,
        mosaic=1.0,
        mixup=0.1,
        project=args.project,
        name=args.name,
        exist_ok=True,
        save=True,
        plots=True,
        device=0 if (model.device and str(model.device) != "cpu") else "cpu"
    )

    # Validation & Performance Metrics Report
    logger.info("Evaluating Tier-1 YOLOv8 model metrics on validation set...")
    metrics = model.val()
    
    logger.info(f"Validation mAP@50    : {metrics.box.map50:.4f}")
    logger.info(f"Validation mAP@50-95 : {metrics.box.map:.4f}")
    logger.info(f"Precision            : {metrics.box.mp:.4f}")
    logger.info(f"Recall               : {metrics.box.mr:.4f}")

    # Export best PyTorch checkpoint to models root
    best_ckpt = Path(args.project) / args.name / "weights" / "best.pt"
    target_ckpt = Path(args.project) / "yolo_person.pt"
    if best_ckpt.exists():
        import shutil
        target_ckpt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_ckpt, target_ckpt)
        logger.info(f"Exported primary Tier-1 model checkpoint -> {target_ckpt}")


if __name__ == "__main__":
    main()
