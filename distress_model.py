#!/usr/bin/env python3
"""
Project Vigilance - Tier 2 In-House Spatio-Temporal Model Core (`distress_model.py`)
Provides standalone access and factory functions for DistressSpatioTemporalNet.
"""

from typing import Dict, Any, Optional
import torch

from src.models.distress_net import DistressSpatioTemporalNet, TemporalSelfAttention, SpatialJointEmbedding


def get_distress_model(
    num_keypoints: int = 33,
    channels: int = 3,
    num_classes: int = 4,
    device: str = "cpu"
) -> DistressSpatioTemporalNet:
    """Factory helper function to instantiate Tier-2 Distress Model."""
    model = DistressSpatioTemporalNet(
        num_keypoints=num_keypoints,
        channels=channels,
        num_classes=num_classes
    )
    return model.to(device)


if __name__ == "__main__":
    # Self-test block
    print("Testing Tier-2 DistressSpatioTemporalNet architecture...")
    dummy_input = torch.randn(8, 30, 33, 3)  # Batch=8, Frames=30, Keypoints=33, Channels=3
    model = get_distress_model(num_keypoints=33, channels=3, num_classes=4)
    logits = model(dummy_input)
    print(f"Input Tensor Shape : {dummy_input.shape}")
    print(f"Output Logits Shape: {logits.shape}")
    print(f"Total Model Params : {model.count_parameters():,}")
    assert logits.shape == (8, 4), "Architecture shape mismatch!"
    print("Self-test PASSED successfully!")
