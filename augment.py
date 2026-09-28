#!/usr/bin/env python3
"""
CLI Tool: Real-World CCTV Data Augmentation Pipeline
Usage:
    python augment.py --mode all --yolo_dir data/yolo_dataset --seq_dir data/sequence_dataset --out_dir data/augmented
"""

import argparse
from pathlib import Path
import sys
import numpy as np

from src.annotation.sequence_builder import SequenceDatasetBuilder
from src.augmentation.image_augmentor import YOLOImageAugmentor
from src.augmentation.sequence_augmentor import SequenceAugmentor
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("AugmentationPipelineCLI")


def main():
    parser = argparse.ArgumentParser(description="Project Vigilance - CCTV Data Augmentation Suite")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--mode", type=str, choices=["all", "yolo", "sequence"], default="all", help="Target dataset to augment")
    parser.add_argument("--yolo_dir", type=str, default=None, help="Input YOLO dataset directory")
    parser.add_argument("--seq_dir", type=str, default=None, help="Input sequence dataset directory")
    parser.add_argument("--out_dir", type=str, default=None, help="Output folder for augmented datasets")

    args = parser.parse_args()

    cfg = load_config(args.config) if Path(args.config).exists() else {}
    aug_cfg = cfg.get("augmentation", {})
    paths_cfg = cfg.get("paths", {})

    out_base = Path(args.out_dir or paths_cfg.get("augmented_dir", "data/augmented"))

    # 1. YOLO Image & Bounding Box Augmentation
    if args.mode in ["all", "yolo"]:
        yolo_in = Path(args.yolo_dir or paths_cfg.get("yolo_dataset_dir", "data/yolo_dataset"))
        yolo_aug_cfg = aug_cfg.get("yolo", {})

        image_aug = YOLOImageAugmentor(
            motion_blur_prob=yolo_aug_cfg.get("motion_blur_prob", 0.4),
            brightness_contrast_prob=yolo_aug_cfg.get("brightness_contrast_prob", 0.5),
            gaussian_noise_prob=yolo_aug_cfg.get("gaussian_noise_prob", 0.3),
            perspective_prob=yolo_aug_cfg.get("perspective_prob", 0.3),
            occlusion_prob=yolo_aug_cfg.get("occlusion_prob", 0.4)
        )

        for split in ["train"]:  # Typically only augment training split
            img_dir = yolo_in / "images" / split
            lbl_dir = yolo_in / "labels" / split
            out_img = out_base / "yolo" / "images" / split
            out_lbl = out_base / "yolo" / "labels" / split

            if img_dir.exists() and lbl_dir.exists():
                logger.info(f"Running YOLO Augmentations for split: {split}...")
                image_aug.augment_dataset(
                    images_dir=img_dir,
                    labels_dir=lbl_dir,
                    out_images_dir=out_img,
                    out_labels_dir=out_lbl,
                    samples_per_image=yolo_aug_cfg.get("samples_per_image", 3)
                )

    # 2. Keypoint Sequence Temporal & Spatial Augmentation
    if args.mode in ["all", "sequence"]:
        seq_in = Path(args.seq_dir or paths_cfg.get("sequence_dataset_dir", "data/sequence_dataset"))
        seq_aug_cfg = aug_cfg.get("sequence", {})

        npy_x_path = seq_in / "X_sequences.npy"
        npy_y_path = seq_in / "y_sequences.npy"

        if npy_x_path.exists() and npy_y_path.exists():
            logger.info("Loading existing keypoint sequence dataset for augmentation...")
            X = np.load(npy_x_path)
            y = np.load(npy_y_path)

            seq_aug = SequenceAugmentor(
                jitter_std=seq_aug_cfg.get("spatial_jitter_std", 0.015),
                scale_range=tuple(seq_aug_cfg.get("spatial_scale_range", [0.85, 1.15])),
                rotation_deg_max=seq_aug_cfg.get("spatial_rotation_deg", 10.0),
                frame_dropout_prob=seq_aug_cfg.get("frame_dropout_prob", 0.3),
                max_dropout_frames=seq_aug_cfg.get("max_dropout_frames", 3),
                speed_stretch_prob=seq_aug_cfg.get("speed_stretch_prob", 0.3)
            )

            X_aug, y_aug = seq_aug.augment_dataset(
                X, y,
                samples_per_sequence=seq_aug_cfg.get("samples_per_sequence", 3)
            )

            out_seq_dir = out_base / "sequences"
            builder = SequenceDatasetBuilder()
            builder.save_dataset(
                X=X_aug,
                y=y_aug,
                output_dir=out_seq_dir,
                h5_filename="augmented_sequences.h5",
                npy_x_name="X_augmented.npy",
                npy_y_name="y_augmented.npy"
            )

    logger.info("=== AUGMENTATION PIPELINE FINISHED ===")


if __name__ == "__main__":
    main()
