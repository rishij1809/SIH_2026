#!/usr/bin/env python3
"""
CLI Tool: Multi-Stream Support & Concurrency Manager (`multi_stream_runner.py`)
Usage:
    python multi_stream_runner.py --config configs/pipeline_config.yaml
"""

import argparse
from pathlib import Path
import time

from src.pipeline.multi_stream_runner import MultiStreamRunner
from src.utils.logger import setup_logger

logger = setup_logger("MultiStreamCLI")


def main():
    parser = argparse.ArgumentParser(description="Project Vigilance - Multi-Stream Surveillance Runner")
    parser.add_argument("--config", type=str, default="configs/pipeline_config.yaml", help="Path to pipeline config YAML")
    parser.add_argument("--duration", type=int, default=15, help="Duration to run in seconds")

    args = parser.parse_args()

    runner = MultiStreamRunner(config_path=args.config)
    runner.start()

    logger.info(f"Multi-Stream runner active. Running for {args.duration} seconds...")

    try:
        start_t = time.time()
        while time.time() - start_t < args.duration:
            time.sleep(1.0)
            status_str = " | ".join([
                f"{w.camera_name}: {w.processed_frames} frames ({w.total_alerts} alerts)"
                for w in runner.workers.values()
            ])
            logger.info(f"STATUS -> {status_str}")
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        runner.stop()

    logger.info("Multi-Stream execution finished successfully.")


if __name__ == "__main__":
    main()
