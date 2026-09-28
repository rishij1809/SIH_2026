"""
Augmentation module for CCTV bounding box images and skeletal keypoint sequences.
"""

from .image_augmentor import YOLOImageAugmentor
from .sequence_augmentor import SequenceAugmentor

__all__ = ["YOLOImageAugmentor", "SequenceAugmentor"]
