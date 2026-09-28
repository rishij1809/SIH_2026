#!/usr/bin/env python3
"""
CLI Tool: Heuristic Verification & Temporal Gating (`verification_engine.py`)
Usage:
    python verification_engine.py
"""

import time
from src.alerts.verification_engine import HeuristicVerificationEngine
from src.utils.logger import setup_logger

logger = setup_logger("VerificationEngineCLI")


def main():
    verifier = HeuristicVerificationEngine(
        confidence_threshold=0.65,
        persistence_window_N=5,
        persistence_threshold_M=3,
        cooldown_sec=10.0
    )

    logger.info("Simulating detection stream for Track #101...")

    bbox = [100.0, 150.0, 200.0, 350.0]

    # Frame 1-2: Single transient noise spikes
    is_ver, reason = verifier.evaluate_detection("CAM_01", 101, 1, 0.70, bbox)
    logger.info(f"Eval 1: Verified={is_ver} | Reason: {reason}")

    is_ver, reason = verifier.evaluate_detection("CAM_01", 101, 1, 0.80, bbox)
    logger.info(f"Eval 2: Verified={is_ver} | Reason: {reason}")

    # Frame 3: 3rd consecutive detection -> Triggers verified incident!
    is_ver, reason = verifier.evaluate_detection("CAM_01", 101, 1, 0.85, bbox)
    logger.info(f"Eval 3: Verified={is_ver} | Reason: {reason}")

    # Frame 4: Immediate duplicate detection -> Suppressed by cooldown!
    is_ver, reason = verifier.evaluate_detection("CAM_01", 101, 1, 0.90, bbox)
    logger.info(f"Eval 4: Verified={is_ver} | Reason: {reason}")

    logger.info("Verification Engine simulation completed successfully!")


if __name__ == "__main__":
    main()
