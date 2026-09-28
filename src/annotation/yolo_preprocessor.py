"""
YOLOv8 Dataset Preprocessor & Splitter for Project Vigilance.
Validates normalized bounding box format, creates train/val/test splits (70/20/10), and auto-generates data.yaml.
"""

import os
from pathlib import Path
import random
import shutil
from typing import Dict, List, Tuple, Union
import yaml

from src.utils.logger import setup_logger

logger = setup_logger("YOLOPreprocessor")


class YOLODataPreprocessor:
    """Preprocesses, validates, splits, and structures YOLO bounding box annotations."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.20,
        test_ratio: float = 0.10,
        random_seed: int = 42,
        class_names: Dict[int, str] = None
    ) -> None:
        """
        Args:
            train_ratio: Fraction for training split.
            val_ratio: Fraction for validation split.
            test_ratio: Fraction for test split.
            random_seed: Seed for reproducible data splitting.
            class_names: Mapping of class index to class name.
        """
        total = train_ratio + val_ratio + test_ratio
        if not abs(total - 1.0) < 1e-4:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.class_names = class_names or {0: "person"}

    @staticmethod
    def validate_yolo_label_file(label_path: Path) -> Tuple[bool, List[str]]:
        """
        Validates whether a label file follows strict YOLO format:
        `class_id x_center y_center width height` with coordinates in [0.0, 1.0].
        """
        errors = []
        if not label_path.exists():
            return False, ["Label file does not exist"]

        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for idx, line in enumerate(lines, 1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                errors.append(f"Line {idx}: Expected 5 elements, got {len(parts)}")
                continue

            try:
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:])
            except ValueError:
                errors.append(f"Line {idx}: Non-numeric values found")
                continue

            if cls_id < 0:
                errors.append(f"Line {idx}: Negative class_id {cls_id}")
            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                errors.append(f"Line {idx}: Out-of-bounds center coordinates ({xc}, {yc})")
            if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                errors.append(f"Line {idx}: Invalid bounding box dimensions ({w}, {h})")

        return len(errors) == 0, errors

    def split_and_structure(
        self,
        images_source_dir: Union[str, Path],
        labels_source_dir: Union[str, Path],
        target_dataset_dir: Union[str, Path],
        yaml_output_path: Union[str, Path] = "configs/data.yaml"
    ) -> Dict[str, int]:
        """
        Splits dataset into train/val/test splits and generates data.yaml.

        Args:
            images_source_dir: Source folder containing image files (.jpg, .png).
            labels_source_dir: Source folder containing YOLO .txt annotations.
            target_dataset_dir: Output root dataset directory.
            yaml_output_path: Path to write auto-generated data.yaml.
        """
        img_dir = Path(images_source_dir)
        lbl_dir = Path(labels_source_dir)
        target_dir = Path(target_dataset_dir)

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files = [f for f in img_dir.glob("*") if f.suffix.lower() in valid_exts]

        if not image_files:
            logger.warning(f"No image files found in {img_dir}")
            return {"train": 0, "val": 0, "test": 0}

        matched_pairs = []
        invalid_count = 0

        for img_path in image_files:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                is_valid, errs = self.validate_yolo_label_file(lbl_path)
                if is_valid:
                    matched_pairs.append((img_path, lbl_path))
                else:
                    logger.warning(f"Invalid label {lbl_path.name}: {errs[:2]}")
                    invalid_count += 1

        logger.info(f"Found {len(matched_pairs)} valid image-label pairs ({invalid_count} invalid skipped).")

        random.seed(self.random_seed)
        random.shuffle(matched_pairs)

        n_total = len(matched_pairs)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)

        splits = {
            "train": matched_pairs[:n_train],
            "val": matched_pairs[n_train:n_train + n_val],
            "test": matched_pairs[n_train + n_val:]
        }

        split_counts = {}
        for split_name, pairs in splits.items():
            img_split_dir = target_dir / "images" / split_name
            lbl_split_dir = target_dir / "labels" / split_name

            img_split_dir.mkdir(parents=True, exist_ok=True)
            lbl_split_dir.mkdir(parents=True, exist_ok=True)

            for img_path, lbl_path in pairs:
                shutil.copy2(img_path, img_split_dir / img_path.name)
                shutil.copy2(lbl_path, lbl_split_dir / lbl_path.name)

            split_counts[split_name] = len(pairs)
            logger.info(f"Split [{split_name.upper()}]: {len(pairs)} pairs copied to {img_split_dir}")

        self.generate_data_yaml(target_dir, yaml_output_path)
        return split_counts

    def generate_data_yaml(self, dataset_dir: Path, yaml_path: Union[str, Path]) -> None:
        """Generates YOLOv8 standard data.yaml dataset specification."""
        out_yaml = Path(yaml_path)
        out_yaml.parent.mkdir(parents=True, exist_ok=True)

        data_dict = {
            "path": str(dataset_dir.resolve()).replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(self.class_names),
            "names": {int(k): str(v) for k, v in self.class_names.items()}
        }

        with open(out_yaml, "w", encoding="utf-8") as f:
            yaml.dump(data_dict, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Generated YOLOv8 config at: {out_yaml}")
