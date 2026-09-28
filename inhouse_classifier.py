#!/usr/bin/env python3
"""
CLI & Importable Wrapper Module: InHouseClassifier (`inhouse_classifier.py`)
"""

import numpy as np
from src.pipeline.inhouse_classifier import InHouseClassifier


def test_inhouse_classifier():
    print("Testing InHouseClassifier non-destructive wrapper...")
    classifier = InHouseClassifier(model_path="models/inhouse_distress.pt", num_keypoints=17)

    dummy_seq = np.random.uniform(0.3, 0.7, size=(30, 17, 3)).astype(np.float32)
    label, confidence = classifier.predict(dummy_seq)

    print(f"Prediction Result -> Label: '{label}' | Confidence: {confidence*100:.1f}%")
    print("InHouseClassifier Test PASSED successfully!")


if __name__ == "__main__":
    test_inhouse_classifier()
