#!/usr/bin/env python3
"""
CLI Tool: Dual-Format Annotation Component A - YOLOv8 Dataset Preprocessor & Splitter
Usage:
    python prepare_yolo.py --images data/extracted_frames --labels data/raw_labels --target data/yolo_dataset
"""

import argparse
from pathlib import Path
import sys

from src.annotation.yolo_preprocessor import YOLODataPreprocessor
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("PrepareYOLO")


def main():
    parser = argparse.ArgumentParser(description="Project Vigilance - YOLOv8 Dataset Structurer")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--images", type=str, default=None, help="Directory containing input images")
    parser.add_argument("--labels", type=str, default=None, help="Directory containing YOLO txt label files")
    parser.add_argument("--target", type=str, default=None, help="Output directory for structured YOLO dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")

    args = parser.parse_args()

    cfg = load_config(args.config) if Path(args.config).exists() else {}
    yolo_cfg = cfg.get("yolo", {})
    paths_cfg = cfg.get("paths", {})

    images_dir = Path(args.images or paths_cfg.get("extracted_frames_dir", "data/extracted_frames"))
    labels_dir = Path(args.labels or (images_dir.parent / "raw_labels"))
    target_dir = Path(args.target or paths_cfg.get("yolo_dataset_dir", "data/yolo_dataset"))
    yaml_out = yolo_cfg.get("yaml_output_path", "configs/data.yaml")

    split_ratios = yolo_cfg.get("split_ratios", {"train": 0.70, "val": 0.20, "test": 0.10})
    classes = yolo_cfg.get("classes", {0: "person"})

    preprocessor = YOLODataPreprocessor(
        train_ratio=split_ratios.get("train", 0.70),
        val_ratio=split_ratios.get("val", 0.20),
        test_ratio=split_ratios.get("test", 0.10),
        random_seed=args.seed,
        class_names=classes
    )

    if not images_dir.exists():
        logger.warning(f"Images folder '{images_dir}' does not exist. Creating directories...")
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        sys.exit(0)

    if not labels_dir.exists():
        logger.warning(f"Labels folder '{labels_dir}' does not exist. Creating dummy sample folder...")
        labels_dir.mkdir(parents=True, exist_ok=True)

    counts = preprocessor.split_and_structure(
        images_source_dir=images_dir,
        labels_source_dir=labels_dir,
        target_dataset_dir=target_dir,
        yaml_output_path=yaml_out
    )

    logger.info(f"YOLO Dataset structuring completed successfully: {counts}")


if __name__ == "__main__":
    main()
