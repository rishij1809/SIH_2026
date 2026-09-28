#!/usr/bin/env python3
"""
CLI Tool: Frame Extraction & CCTV Ingestion Tool
Usage:
    python extract_frames.py --input data/raw_videos --output data/extracted_frames --fps 12 --method ssim --threshold 0.94
"""

import argparse
from pathlib import Path
import sys

from src.ingestion.frame_extractor import FrameExtractor
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("ExtractFramesCLI")


def main():
    parser = argparse.ArgumentParser(description="Project Vigilance - CCTV Frame Extractor")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--input", type=str, default=None, help="Path to raw video file or folder containing videos")
    parser.add_argument("--output", type=str, default=None, help="Output directory for extracted frames")
    parser.add_argument("--fps", type=float, default=None, help="Target sampling rate FPS")
    parser.add_argument("--method", type=str, choices=["ssim", "diff"], default=None, help="Similarity metric")
    parser.add_argument("--threshold", type=float, default=None, help="Duplicate similarity threshold [0.0-1.0]")
    
    args = parser.parse_args()

    cfg = load_config(args.config) if Path(args.config).exists() else {}
    ingest_cfg = cfg.get("ingestion", {})
    paths_cfg = cfg.get("paths", {})

    input_path = Path(args.input or paths_cfg.get("raw_videos_dir", "data/raw_videos"))
    output_path = Path(args.output or paths_cfg.get("extracted_frames_dir", "data/extracted_frames"))
    target_fps = args.fps or ingest_cfg.get("target_fps", 12.0)
    method = args.method or ingest_cfg.get("similarity_method", "ssim")
    threshold = args.threshold or ingest_cfg.get("similarity_threshold", 0.94)

    extractor = FrameExtractor(
        target_fps=target_fps,
        similarity_method=method,
        similarity_threshold=threshold,
        resize_dim=(ingest_cfg.get("resize_width", 1280), ingest_cfg.get("resize_height", 720)),
        save_format=ingest_cfg.get("save_format", "png")
    )

    if not input_path.exists():
        logger.warning(f"Input path '{input_path}' does not exist. Creating directory and exiting...")
        input_path.mkdir(parents=True, exist_ok=True)
        sys.exit(0)

    video_files = []
    if input_path.is_file():
        video_files.append(input_path)
    else:
        for ext in ["*.mp4", "*.avi", "*.mkv", "*.mov", "*.flv"]:
            video_files.extend(list(input_path.glob(ext)))
            video_files.extend(list(input_path.glob(f"**/{ext}")))

    if not video_files:
        logger.warning(f"No video files found in {input_path}")
        return

    logger.info(f"Found {len(video_files)} video file(s) for extraction.")
    total_stats = {"total_frames_read": 0, "saved_frames": 0, "skipped_fps_frames": 0, "skipped_duplicate_frames": 0}

    for vid in video_files:
        stats = extractor.process_video(video_source=vid, output_dir=output_path)
        for k in total_stats:
            total_stats[k] += stats.get(k, 0)

    logger.info(
        f"=== INGESTION SUMMARY ===\n"
        f"Videos Processed: {len(video_files)}\n"
        f"Total Raw Frames Read: {total_stats['total_frames_read']}\n"
        f"Saved Frames: {total_stats['saved_frames']}\n"
        f"Skipped (FPS Subsampling): {total_stats['skipped_fps_frames']}\n"
        f"Skipped (SSIM/Diff Duplicates): {total_stats['skipped_duplicate_frames']}"
    )


if __name__ == "__main__":
    main()
