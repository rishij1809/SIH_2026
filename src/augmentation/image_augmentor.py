"""
YOLO Image & Bounding Box Augmentor using Albumentations.
Simulates motion blur, low-light night noise, Gaussian noise, perspective distortion, and partial occlusions.
Preserves normalized YOLO format [x_center, y_center, width, height].
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union
import albumentations as A
import cv2
import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger("YOLOImageAugmentor")


class YOLOImageAugmentor:
    """Applies realistic CCTV environment augmentations to images and YOLO bounding boxes."""

    def __init__(
        self,
        motion_blur_prob: float = 0.4,
        brightness_contrast_prob: float = 0.5,
        gaussian_noise_prob: float = 0.3,
        perspective_prob: float = 0.3,
        occlusion_prob: float = 0.4
    ) -> None:
        """Configures Albumentations pipeline targeting low-resource surveillance feeds."""
        self.transform = A.Compose(
            [
                # CCTV Motion Blur
                A.MotionBlur(blur_limit=(3, 11), p=motion_blur_prob),
                # Low light / night grain / camera sensor noise
                A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.2), contrast_limit=(-0.3, 0.3), p=brightness_contrast_prob),
                A.GaussNoise(var_limit=(10.0, 50.0), p=gaussian_noise_prob),
                # Camera perspective / angle shift distortion
                A.Perspective(scale=(0.02, 0.08), p=perspective_prob),
                # Partial occlusions (simulating poles, passing crowds, shadows)
                A.CoarseDropout(
                    max_holes=6,
                    max_height=32,
                    max_width=32,
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=occlusion_prob
                ),
            ],
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
                min_visibility=0.25
            )
        )

    def augment_sample(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        class_labels: List[int]
    ) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """
        Augments a single image and set of normalized YOLO bounding boxes.

        Args:
            image: BGR input frame array.
            bboxes: List of [x_center, y_center, width, height] normalized coordinates.
            class_labels: List of integer class IDs.

        Returns:
            Tuple of (augmented_image, augmented_bboxes, augmented_class_labels).
        """
        if not bboxes:
            # Image has no objects
            augmented = self.transform(image=image, bboxes=[], class_labels=[])
            return augmented["image"], [], []

        try:
            res = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            return res["image"], res["bboxes"], res["class_labels"]
        except Exception as e:
            logger.debug(f"Augmentation failed for image (retaining original): {e}")
            return image, bboxes, class_labels

    def augment_dataset(
        self,
        images_dir: Union[str, Path],
        labels_dir: Union[str, Path],
        out_images_dir: Union[str, Path],
        out_labels_dir: Union[str, Path],
        samples_per_image: int = 3
    ) -> int:
        """
        Reads YOLO image-label dataset, applies augmentations, and writes outputs.
        """
        img_path = Path(images_dir)
        lbl_path = Path(labels_dir)
        out_img_path = Path(out_images_dir)
        out_lbl_path = Path(out_labels_dir)

        out_img_path.mkdir(parents=True, exist_ok=True)
        out_lbl_path.mkdir(parents=True, exist_ok=True)

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files = [f for f in img_path.glob("*") if f.suffix.lower() in valid_exts]

        total_generated = 0
        for img_f in image_files:
            lbl_f = lbl_path / f"{img_f.stem}.txt"
            if not lbl_f.exists():
                continue

            img = cv2.imread(str(img_f))
            if img is None:
                continue

            bboxes = []
            classes = []
            with open(lbl_f, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        classes.append(int(parts[0]))
                        bboxes.append([float(x) for x in parts[1:]])

            for i in range(samples_per_image):
                aug_img, aug_boxes, aug_cls = self.augment_sample(img, bboxes, classes)
                
                new_stem = f"{img_f.stem}_aug_{i+1}"
                cv2.imwrite(str(out_img_path / f"{new_stem}{img_f.suffix}"), aug_img)

                with open(out_lbl_path / f"{new_stem}.txt", "w") as f_out:
                    for c, box in zip(aug_cls, aug_boxes):
                        f_out.write(f"{c} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")
                
                total_generated += 1

        logger.info(f"YOLO Augmentation finished: Generated {total_generated} augmented samples.")
        return total_generated
