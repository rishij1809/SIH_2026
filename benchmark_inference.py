#!/usr/bin/env python3
"""
CLI Tool: Model Verification & Operational Benchmark Script (`benchmark_inference.py`)
Passes dummy batch simulating real-time CCTV inputs through Tier-1 & Tier-2 models.
Outputs operational profile table (Parameters, FLOPs, Latency ms, FPS).
"""

import argparse
from pathlib import Path
import sys
import torch

from src.models.distress_net import DistressSpatioTemporalNet
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.validation.benchmark import OperationalBenchmark

logger = setup_logger("BenchmarkCLI")


def main():
    parser = argparse.ArgumentParser(description="Project Vigilance - Edge Model Latency & Operational Profiler")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--iters", type=int, default=100, help="Number of benchmark iterations")

    args = parser.parse_args()

    bench = OperationalBenchmark()

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    profile_results = []

    # 1. Tier 2: In-House Spatio-Temporal Distress Model Benchmark
    logger.info("Profiling Tier-2 In-House Spatio-Temporal AI Model...")
    distress_model = DistressSpatioTemporalNet(num_keypoints=33, channels=3, num_classes=4)
    dummy_distress_input = torch.randn(1, 30, 33, 3)

    for dev in devices:
        res = bench.benchmark_model_latency(
            model=distress_model,
            dummy_input=dummy_distress_input,
            device_str=dev,
            num_iters=args.iters
        )
        res["model_name"] = "Tier-2 Spatio-Temporal Model"
        profile_results.append(res)

    # 2. Tier 1: YOLOv8 Spatial Localization Benchmark (if Ultralytics available)
    try:
        from ultralytics import YOLO
        logger.info("Profiling Tier-1 YOLOv8 Person Localization Model...")
        yolo_path = Path("models/yolo_person.pt")
        base_yolo = "yolov8n.pt" if not yolo_path.exists() else str(yolo_path)
        
        yolo_model = YOLO(base_yolo).model
        dummy_yolo_input = torch.randn(1, 3, 640, 640)

        for dev in devices:
            res = bench.benchmark_model_latency(
                model=yolo_model,
                dummy_input=dummy_yolo_input,
                device_str=dev,
                num_iters=args.iters
            )
            res["model_name"] = f"Tier-1 YOLOv8 ({Path(base_yolo).stem})"
            profile_results.append(res)

    except Exception as e:
        logger.warning(f"Skipping YOLO profiling due to: {e}")

    # Output Operational Report
    bench.print_operational_profile(profile_results)


if __name__ == "__main__":
    main()
