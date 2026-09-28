#!/usr/bin/env python3
"""
Weapon (Knife) Dataset Preparation & Augmentation Script
Copies user-provided knife photos, generates bounding box annotations,
and augments dataset into YOLO structure.
"""

from pathlib import Path
import shutil
import cv2
import numpy as np

def main():
    base_dir = Path("data/knife_dataset")
    img_dir = base_dir / "images"
    lbl_dir = base_dir / "labels"
    
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # 5 User Knife Images
    source_files = [
        ("WhatsApp Image 2026-09-28 at 2.46.37 AM (1).jpeg", [0, 0.49, 0.47, 0.16, 0.72]),
        ("WhatsApp Image 2026-09-28 at 2.46.37 AM (2).jpeg", [0, 0.46, 0.51, 0.62, 0.14]),
        ("WhatsApp Image 2026-09-28 at 2.46.37 AM (3).jpeg", [0, 0.48, 0.52, 0.65, 0.14]),
        ("WhatsApp Image 2026-09-28 at 2.46.37 AM.jpeg",     [0, 0.51, 0.48, 0.16, 0.70]),
        ("WhatsApp Image 2026-09-28 at 2.46.38 AM.jpeg",     [0, 0.49, 0.51, 0.68, 0.14]),
    ]

    print("Copying 5 user knife images and generating YOLO bounding box annotations...")
    valid_count = 0
    for idx, (filename, bbox) in enumerate(source_files, 1):
        src_path = Path(filename)
        if not src_path.exists():
            print(f"File not found: {filename}")
            continue

        target_img = img_dir / f"knife_{idx:02d}.jpg"
        target_lbl = lbl_dir / f"knife_{idx:02d}.txt"

        shutil.copy2(src_path, target_img)

        # Write YOLO label: class_id x_center y_center width height
        with open(target_lbl, "w") as f:
            f.write(f"{bbox[0]} {bbox[1]:.4f} {bbox[2]:.4f} {bbox[3]:.4f} {bbox[4]:.4f}\n")

        valid_count += 1
        print(f"Processed: {target_img.name} -> {target_lbl.name}")

    print(f"Successfully staged {valid_count} baseline knife images.")

    # Generate 100 Augmented Knife Samples
    out_yolo = Path("data/yolo_dataset")
    train_img_dir = out_yolo / "images" / "train"
    train_lbl_dir = out_yolo / "labels" / "train"
    val_img_dir = out_yolo / "images" / "val"
    val_lbl_dir = out_yolo / "labels" / "val"

    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_lbl_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    aug_count = 0
    raw_images = list(img_dir.glob("*.jpg"))

    for img_path in raw_images:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        with open(lbl_path, "r") as f:
            line = f.readline().strip()
            parts = line.split()
            cls_id, xc, yc, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

        # Copy original
        shutil.copy2(img_path, train_img_dir / img_path.name)
        shutil.copy2(lbl_path, train_lbl_dir / (img_path.stem + ".txt"))
        shutil.copy2(img_path, val_img_dir / img_path.name)
        shutil.copy2(lbl_path, val_lbl_dir / (img_path.stem + ".txt"))
        aug_count += 1

        # Generate 20 Augmented Variations per image (rotations, flips, brightness, scales)
        for i in range(20):
            aug_img = img.copy()
            new_xc, new_yc, new_bw, new_bh = xc, yc, bw, bh

            # Random Brightness & Contrast
            alpha = np.random.uniform(0.7, 1.3)
            beta = np.random.randint(-30, 30)
            aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)

            # Random Horizontal Flip
            if np.random.rand() > 0.5:
                aug_img = cv2.flip(aug_img, 1)
                new_xc = 1.0 - xc

            # Random Vertical Flip
            if np.random.rand() > 0.5:
                aug_img = cv2.flip(aug_img, 0)
                new_yc = 1.0 - yc

            aug_filename = f"{img_path.stem}_aug_{i:02d}.jpg"
            cv2.imwrite(str(train_img_dir / aug_filename), aug_img)
            with open(train_lbl_dir / (f"{img_path.stem}_aug_{i:02d}.txt"), "w") as f:
                f.write(f"{cls_id} {new_xc:.4f} {new_yc:.4f} {new_bw:.4f} {new_bh:.4f}\n")

            aug_count += 1

    print(f"=== KNIFE DATASET AUGMENTED ===")
    print(f"Total training & validation samples: {aug_count}")

    # Generate configs/data.yaml
    config_yaml = Path("configs/data.yaml")
    config_yaml.parent.mkdir(parents=True, exist_ok=True)
    yaml_content = f"""path: {out_yolo.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: knife
  1: person
"""
    with open(config_yaml, "w") as f:
        f.write(yaml_content)

    print(f"Generated YOLO config: {config_yaml}")

if __name__ == "__main__":
    main()
