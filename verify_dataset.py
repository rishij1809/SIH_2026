#!/usr/bin/env python3
"""
CLI Tool: Data Validation & Integrity Checker
Usage:
    python verify_dataset.py --yolo_dir data/yolo_dataset --seq_dir data/sequence_dataset
"""

import argparse
from pathlib import Path
import sys

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.validation.dataset_verifier import DatasetVerifier

logger = setup_logger("VerifyDatasetCLI")


def main():
    parser = argparse.ArgumentParser(description="Project Vigilance - Dataset Verification Suite")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--yolo_dir", type=str, default=None, help="YOLO dataset root folder")
    parser.add_argument("--seq_dir", type=str, default=None, help="Sequence dataset root folder")
    parser.add_argument("--imbalance_threshold", type=float, default=3.0, help="Max allowed class imbalance ratio")

    args = parser.parse_args()

    cfg = load_config(args.config) if Path(args.config).exists() else {}
    paths_cfg = cfg.get("paths", {})

    yolo_dir = Path(args.yolo_dir or paths_cfg.get("yolo_dataset_dir", "data/yolo_dataset"))
    seq_dir = Path(args.seq_dir or paths_cfg.get("sequence_dataset_dir", "data/sequence_dataset"))

    verifier = DatasetVerifier(warning_imbalance_ratio=args.imbalance_threshold)

    logger.info("Starting Dataset Verification...")
    yolo_report = verifier.verify_yolo_dataset(yolo_dir)

    npy_x = seq_dir / "X_sequences.npy"
    npy_y = seq_dir / "y_sequences.npy"
    seq_report = verifier.verify_sequence_dataset(npy_x, npy_y)

    verifier.print_summary_report(yolo_report, seq_report)


if __name__ == "__main__":
    main()
