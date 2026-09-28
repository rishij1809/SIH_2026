"""
CLI Verification Utility for Project Vigilance Dataset Integrity.
Validates missing files, image corruption, bounding box coordinates, class imbalance, and keypoint sequence continuity.
"""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import cv2
import h5py
import numpy as np
from tabulate import tabulate

from src.utils.logger import setup_logger

logger = setup_logger("DatasetVerifier")


class DatasetVerifier:
    """Verifies dataset health, missing labels, corrupted frames, coordinate bounds, and sequence continuity."""

    def __init__(self, warning_imbalance_ratio: float = 3.0) -> None:
        self.warning_imbalance_ratio = warning_imbalance_ratio

    def verify_image_file(self, img_path: Path) -> Tuple[bool, str]:
        """Checks whether image file is readable and non-empty."""
        if not img_path.exists():
            return False, "File does not exist"
        if img_path.stat().st_size == 0:
            return False, "File size is 0 bytes"
        try:
            img = cv2.imread(str(img_path))
            if img is None or img.size == 0:
                return False, "Corrupted/Unreadable image binary"
            return True, "Valid"
        except Exception as e:
            return False, f"OpenCV Exception: {str(e)}"

    def verify_yolo_dataset(self, yolo_root: Union[str, Path]) -> Dict[str, Any]:
        """
        Validates YOLO images/labels directory hierarchy across train/val/test splits.
        Checks missing labels, orphan images/labels, corrupted image files, bounding box limits, and class distribution.
        """
        root = Path(yolo_root)
        report = {
            "splits": {},
            "total_images": 0,
            "total_labels": 0,
            "corrupted_images": [],
            "orphaned_images": [],
            "orphaned_labels": [],
            "bbox_errors": [],
            "class_distribution": Counter(),
            "status": "PASSED"
        }

        splits = ["train", "val", "test"]
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

        for split in splits:
            img_dir = root / "images" / split
            lbl_dir = root / "labels" / split

            split_report = {"images": 0, "labels": 0, "valid_pairs": 0, "corrupt": 0}
            
            if not img_dir.exists() or not lbl_dir.exists():
                logger.warning(f"Split path missing: {img_dir} or {lbl_dir}")
                continue

            images = {f.stem: f for f in img_dir.glob("*") if f.suffix.lower() in valid_exts}
            labels = {f.stem: f for f in lbl_dir.glob("*.txt")}

            split_report["images"] = len(images)
            split_report["labels"] = len(labels)
            report["total_images"] += len(images)
            report["total_labels"] += len(labels)

            # Check image readability & matching labels
            for stem, img_path in images.items():
                is_valid, msg = self.verify_image_file(img_path)
                if not is_valid:
                    report["corrupted_images"].append(f"{img_path} ({msg})")
                    split_report["corrupt"] += 1

                if stem not in labels:
                    report["orphaned_images"].append(str(img_path))
                else:
                    lbl_path = labels[stem]
                    # Parse bounding box validity
                    with open(lbl_path, "r", encoding="utf-8") as f:
                        for line_no, line in enumerate(f, 1):
                            parts = line.strip().split()
                            if len(parts) == 5:
                                try:
                                    c = int(parts[0])
                                    xc, yc, w, h = map(float, parts[1:])
                                    report["class_distribution"][c] += 1
                                    
                                    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                                        report["bbox_errors"].append(
                                            f"{lbl_path.name}:L{line_no} Out of bounds box: {[xc, yc, w, h]}"
                                        )
                                except ValueError:
                                    report["bbox_errors"].append(f"{lbl_path.name}:L{line_no} Invalid numeric format")

            for stem, lbl_path in labels.items():
                if stem not in images:
                    report["orphaned_labels"].append(str(lbl_path))

            split_report["valid_pairs"] = len(set(images.keys()) & set(labels.keys())) - split_report["corrupt"]
            report["splits"][split] = split_report

        if report["corrupted_images"] or report["bbox_errors"] or len(report["orphaned_images"]) > 0:
            report["status"] = "WARNINGS / ERRORS DETECTED"

        return report

    def verify_sequence_dataset(
        self,
        npy_x_path: Union[str, Path],
        npy_y_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Validates keypoint sequence tensor continuity, checking NaNs/Infs, dimensions, visibility, and class balance.
        """
        px = Path(npy_x_path)
        py = Path(npy_y_path)

        report = {
            "exists": False,
            "shape_X": None,
            "shape_y": None,
            "nan_count": 0,
            "inf_count": 0,
            "class_counts": {},
            "mean_visibility": 0.0,
            "status": "PASSED"
        }

        if not px.exists() or not py.exists():
            report["status"] = "FILE NOT FOUND"
            return report

        try:
            X = np.load(px)
            y = np.load(py)
        except Exception as e:
            report["status"] = f"LOAD ERROR: {str(e)}"
            return report

        report["exists"] = True
        report["shape_X"] = X.shape
        report["shape_y"] = y.shape

        # Tensor shape check: expected (N, T, V, C)
        if len(X.shape) != 4:
            report["status"] = f"INVALID TENSOR SHAPE: expected 4D (N, T, V, C), got {len(X.shape)}D"

        report["nan_count"] = int(np.isnan(X).sum())
        report["inf_count"] = int(np.isinf(X).sum())

        if report["nan_count"] > 0 or report["inf_count"] > 0:
            report["status"] = "FAILED (NaN or Inf values in keypoint tensor)"

        # Class balance check
        unique_classes, counts = np.unique(y, return_counts=True)
        report["class_counts"] = {int(k): int(v) for k, v in zip(unique_classes, counts)}

        # Mean keypoint visibility (channel index 2 if present)
        if X.shape[-1] >= 3:
            report["mean_visibility"] = float(np.mean(X[:, :, :, 2]))

        return report

    def print_summary_report(self, yolo_report: Dict[str, Any], seq_report: Dict[str, Any]) -> None:
        """Prints rich ASCII table formatted integrity report."""
        print("\n" + "=" * 80)
        print("               PROJECT VIGILANCE - DATASET INTEGRITY REPORT             ")
        print("=" * 80)

        # 1. YOLO Detection Dataset Summary Table
        print("\n[1] YOLOv8 Detection Dataset Integrity:")
        split_table = []
        for split, metrics in yolo_report.get("splits", {}).items():
            split_table.append([split.upper(), metrics["images"], metrics["labels"], metrics["valid_pairs"], metrics["corrupt"]])
        
        print(tabulate(split_table, headers=["Split", "Images", "Labels", "Valid Pairs", "Corrupt"], tablefmt="fancy_grid"))
        print(f"Total Images: {yolo_report.get('total_images', 0)} | Total Labels: {yolo_report.get('total_labels', 0)}")
        print(f"Corrupted Images: {len(yolo_report.get('corrupted_images', []))}")
        print(f"Orphaned Images (Missing Label): {len(yolo_report.get('orphaned_images', []))}")
        print(f"Orphaned Labels (Missing Image): {len(yolo_report.get('orphaned_labels', []))}")
        print(f"BBox Errors: {len(yolo_report.get('bbox_errors', []))}")

        # YOLO Class Distribution
        if yolo_report.get("class_distribution"):
            print("\nYOLO Class Distribution:")
            cls_table = [[f"Class {c}", count] for c, count in yolo_report["class_distribution"].items()]
            print(tabulate(cls_table, headers=["Class ID", "Bounding Box Count"], tablefmt="grid"))

        # 2. Keypoint Sequence Dataset Summary Table
        print("\n[2] Sequential Keypoint Dataset (ST-GCN / Bi-LSTM) Integrity:")
        print(f"File Status: {seq_report.get('status')}")
        if seq_report.get("exists"):
            seq_table = [
                ["X Tensor Shape (N, T, V, C)", str(seq_report["shape_X"])],
                ["y Labels Shape (N,)", str(seq_report["shape_y"])],
                ["NaN / Inf Count", f"{seq_report['nan_count']} / {seq_report['inf_count']}"],
                ["Mean Keypoint Visibility", f"{seq_report['mean_visibility']:.4f}"]
            ]
            print(tabulate(seq_table, headers=["Metric", "Value"], tablefmt="fancy_grid"))

            print("\nSequence Class Balance:")
            lbl_map = {0: "Normal", 1: "Universal Help Signal", 2: "Aggression/Struggle", 3: "Fall"}
            class_table = [
                [c, lbl_map.get(c, f"Class {c}"), count]
                for c, count in seq_report["class_counts"].items()
            ]
            print(tabulate(class_table, headers=["ID", "Gesture Class Name", "Samples"], tablefmt="grid"))

            # Imbalance warning test
            if len(seq_report["class_counts"]) > 1:
                max_c = max(seq_report["class_counts"].values())
                min_c = min(seq_report["class_counts"].values())
                ratio = max_c / max(1, min_c)
                if ratio > self.warning_imbalance_ratio:
                    logger.warning(
                        f"Class Imbalance Warning: Max/Min class sample ratio is {ratio:.2f} (> threshold {self.warning_imbalance_ratio})"
                    )

        print("\n" + "=" * 80 + "\n")
