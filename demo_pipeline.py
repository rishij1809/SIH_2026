#!/usr/bin/env python3
"""
Project Vigilance - Stage 1 End-to-End Pipeline Demonstrator
Executes the full pipeline:
1. Simulates CCTV frame extraction
2. Builds YOLO detection dataset structure & data.yaml
3. Builds Sequential keypoint dataset (X_sequences.npy, y_sequences.npy, sequences.h5)
4. Executes Augmentation Pipeline for YOLO bboxes and keypoint sequences
5. Runs verification suite and renders dataset integrity report
"""

import os
from pathlib import Path
import shutil
import cv2
import numpy as np

from src.annotation.sequence_builder import SequenceDatasetBuilder
from src.annotation.yolo_preprocessor import YOLODataPreprocessor
from src.augmentation.image_augmentor import YOLOImageAugmentor
from src.augmentation.sequence_augmentor import SequenceAugmentor
from src.ingestion.frame_extractor import FrameExtractor
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.validation.dataset_verifier import DatasetVerifier

logger = setup_logger("Stage1DemoPipeline")


def create_demo_assets(base_dir: Path):
    """Creates sample synthetic video frames and YOLO labels for end-to-end pipeline demonstration."""
    raw_vid_dir = base_dir / "data" / "raw_videos"
    extracted_dir = base_dir / "data" / "extracted_frames"
    raw_labels_dir = base_dir / "data" / "raw_labels"

    raw_vid_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    raw_labels_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate synthetic MP4 video
    demo_video_path = raw_vid_dir / "demo_cctv_feed.mp4"
    logger.info(f"Creating synthetic demonstration video at: {demo_video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(demo_video_path), fourcc, 30.0, (1280, 720))
    for t in range(150):
        # Create CCTV frame with moving synthetic figure box
        frame = np.full((720, 1280, 3), (40 + (t % 30), 40, 40), dtype=uint8 if 'uint8' in globals() else np.uint8)
        x_center = int(300 + t * 4)
        y_center = int(400 + np.sin(t * 0.1) * 50)
        cv2.rectangle(frame, (x_center - 40, y_center - 100), (x_center + 40, y_center + 100), (0, 255, 120), -1)
        cv2.putText(frame, "CAM-01 CCTV FEED", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        out.write(frame)
    out.release()

    return demo_video_path


def main():
    base_dir = Path(__file__).parent.resolve()
    logger.info("Initializing Stage 1 Pipeline Execution...")

    # Step 0: Generate baseline demo assets if needed
    demo_video = create_demo_assets(base_dir)

    # Step 1: Ingestion & Dynamic Frame Extraction
    logger.info("\n--- STEP 1: FRAME EXTRACTION & DUPLICATE FILTERING ---")
    extractor = FrameExtractor(target_fps=12.0, similarity_method="ssim", similarity_threshold=0.94)
    extracted_frames_dir = base_dir / "data" / "extracted_frames"
    extractor.process_video(video_source=demo_video, output_dir=extracted_frames_dir)

    # Create dummy matching YOLO label files for extracted frames
    raw_labels_dir = base_dir / "data" / "raw_labels"
    for img_f in extracted_frames_dir.glob("*.png"):
        lbl_f = raw_labels_dir / f"{img_f.stem}.txt"
        with open(lbl_f, "w") as f:
            f.write("0 0.520000 0.550000 0.150000 0.450000\n")

    # Step 2: YOLO Dataset Preprocessing & Splitting (70/20/10)
    logger.info("\n--- STEP 2: YOLO DATASET PREPROCESSING & DATA.YAML ---")
    preprocessor = YOLODataPreprocessor(train_ratio=0.70, val_ratio=0.20, test_ratio=0.10)
    yolo_target_dir = base_dir / "data" / "yolo_dataset"
    preprocessor.split_and_structure(
        images_source_dir=extracted_frames_dir,
        labels_source_dir=raw_labels_dir,
        target_dataset_dir=yolo_target_dir,
        yaml_output_path=base_dir / "configs" / "data.yaml"
    )

    # Step 3: Sequential Keypoint Dataset Building
    logger.info("\n--- STEP 3: SEQUENTIAL KEYPOINT DATASET BUILDING ---")
    builder = SequenceDatasetBuilder(window_size=30, step_size=5, num_keypoints=33, channels=3)
    X, y = builder.generate_synthetic_demo_sequences(num_samples_per_class=30)
    seq_out_dir = base_dir / "data" / "sequence_dataset"
    builder.save_dataset(X, y, output_dir=seq_out_dir)

    # Step 4: Real-World CCTV Data Augmentation
    logger.info("\n--- STEP 4: DATA AUGMENTATION SUITE ---")
    img_aug = YOLOImageAugmentor()
    aug_out_dir = base_dir / "data" / "augmented"
    img_aug.augment_dataset(
        images_dir=yolo_target_dir / "images" / "train",
        labels_dir=yolo_target_dir / "labels" / "train",
        out_images_dir=aug_out_dir / "yolo" / "images" / "train",
        out_labels_dir=aug_out_dir / "yolo" / "labels" / "train",
        samples_per_image=2
    )

    seq_aug = SequenceAugmentor()
    X_aug, y_aug = seq_aug.augment_dataset(X, y, samples_per_sequence=2)
    builder.save_dataset(
        X_aug, y_aug,
        output_dir=aug_out_dir / "sequences",
        h5_filename="augmented_sequences.h5",
        npy_x_name="X_augmented.npy",
        npy_y_name="y_augmented.npy"
    )

    # Step 5: Data Integrity & Verification Checker
    logger.info("\n--- STEP 5: DATASET INTEGRITY & VALIDATION CHECK ---")
    verifier = DatasetVerifier()
    yolo_report = verifier.verify_yolo_dataset(yolo_target_dir)
    seq_report = verifier.verify_sequence_dataset(
        seq_out_dir / "X_sequences.npy",
        seq_out_dir / "y_sequences.npy"
    )
    verifier.print_summary_report(yolo_report, seq_report)

    logger.info("Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
