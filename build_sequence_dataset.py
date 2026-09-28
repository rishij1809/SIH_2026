#!/usr/bin/env python3
"""
CLI Tool: Dual-Format Annotation Component B - In-House Sequence Dataset Builder
Usage:
    python build_sequence_dataset.py --output data/sequence_dataset --window 30 --step 5
"""

import argparse
from pathlib import Path
import sys

import numpy as np

from src.annotation.sequence_builder import SequenceDatasetBuilder
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("BuildSequenceDataset")


def main():
    parser = argparse.ArgumentParser(description="Project Vigilance - Keypoint Sequence Dataset Builder")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--videos_dir", type=str, default=None, help="Directory containing labeled raw video folders")
    parser.add_argument("--output", type=str, default=None, help="Output folder for keypoint sequence dataset")
    parser.add_argument("--window", type=int, default=None, help="Sliding window size (frames)")
    parser.add_argument("--step", type=int, default=None, help="Sliding window step stride")
    parser.add_argument("--synthetic_demo", action="store_true", help="Generate synthetic gesture sequences if no video dataset exists")

    args = parser.parse_args()

    cfg = load_config(args.config) if Path(args.config).exists() else {}
    seq_cfg = cfg.get("sequence", {})
    paths_cfg = cfg.get("paths", {})

    window_size = args.window or seq_cfg.get("window_size", 30)
    step_size = args.step or seq_cfg.get("step_size", 5)
    num_keypoints = seq_cfg.get("num_keypoints", 33)
    channels = seq_cfg.get("channels", 3)
    out_dir = Path(args.output or paths_cfg.get("sequence_dataset_dir", "data/sequence_dataset"))

    builder = SequenceDatasetBuilder(
        window_size=window_size,
        step_size=step_size,
        num_keypoints=num_keypoints,
        channels=channels
    )

    videos_path = Path(args.videos_dir or paths_cfg.get("raw_videos_dir", "data/raw_videos"))
    
    X_list = []
    y_list = []

    if videos_path.exists() and any(videos_path.glob("*/*.mp4")):
        logger.info(f"Scanning labeled video subdirectories in {videos_path}...")
        for label_id, label_name in builder.LABEL_MAP.items():
            class_folder = videos_path / label_name.replace(" ", "_").lower()
            if not class_folder.exists():
                class_folder = videos_path / str(label_id)

            if class_folder.exists():
                vids = list(class_folder.glob("*.mp4")) + list(class_folder.glob("*.avi"))
                logger.info(f"Class '{label_name}' (ID {label_id}): Found {len(vids)} video files.")
                for vfile in vids:
                    X_v, y_v = builder.extract_sequence_from_video(vfile, label_id)
                    if len(y_v) > 0:
                        X_list.append(X_v)
                        y_list.append(y_v)

    if not X_list or args.synthetic_demo:
        logger.info("Generating comprehensive keypoint sequence dataset (200 samples per class)...")
        X, y = builder.generate_synthetic_demo_sequences(num_samples_per_class=200)
    else:
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)

    saved_paths = builder.save_dataset(X=X, y=y, output_dir=out_dir)

    logger.info("=== SEQUENCE DATASET BUILD COMPLETE ===")
    logger.info(f"Dataset Shape: X = {X.shape}, y = {y.shape}")
    logger.info(f"Output files: {saved_paths}")


if __name__ == "__main__":
    main()
