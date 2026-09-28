#!/usr/bin/env python3
"""
CLI Tool: Tier-2 In-House Spatio-Temporal Model Trainer (`train_distress.py`)
Usage:
    python train_distress.py --dataset data/sequence_dataset --epochs 40 --batch_size 32 --lr 1e-3
"""

import argparse
from pathlib import Path
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from torch.utils.data import DataLoader

from src.annotation.sequence_builder import SequenceDatasetBuilder
from src.models.distress_net import DistressSpatioTemporalNet
from src.training.distress_trainer import DistressModelTrainer, KeypointSequenceDataset
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("TrainDistress")


def main():
    parser = argparse.ArgumentParser(description="Project Vigilance - Tier 2 Distress Model Training")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--dataset", type=str, default=None, help="Directory containing sequence dataset (X_sequences.npy, y_sequences.npy)")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="DataLoader batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--focal_loss", action="store_true", default=True, help="Use Focal Loss for class imbalance")
    parser.add_argument("--output", type=str, default="models/distress_model.pt", help="Output model checkpoint path")

    args = parser.parse_args()

    cfg = load_config(args.config) if Path(args.config).exists() else {}
    paths_cfg = cfg.get("paths", {})
    seq_cfg = cfg.get("sequence", {})

    seq_dir = Path(args.dataset or paths_cfg.get("sequence_dataset_dir", "data/sequence_dataset"))
    npy_x = seq_dir / "X_sequences.npy"
    npy_y = seq_dir / "y_sequences.npy"

    if not npy_x.exists() or not npy_y.exists():
        logger.warning(f"Sequence files not found in {seq_dir}. Generating synthetic demonstration dataset...")
        builder = SequenceDatasetBuilder(window_size=30, step_size=5, num_keypoints=33, channels=3)
        X, y = builder.generate_synthetic_demo_sequences(num_samples_per_class=40)
        builder.save_dataset(X, y, output_dir=seq_dir)
    else:
        X = np.load(npy_x)
        y = np.load(npy_y)

    logger.info(f"Loaded Sequence Dataset: X shape = {X.shape}, y shape = {y.shape}")

    # Stratified 80/20 Train/Val Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    logger.info(f"Dataset Split: Train={X_train.shape[0]} samples | Val={X_val.shape[0]} samples")

    # Compute class weights inverse to frequency
    classes, counts = np.unique(y_train, return_counts=True)
    class_weights_np = sum(counts) / (len(classes) * counts.astype(np.float32))
    class_weights_tensor = torch.from_numpy(class_weights_np).float()
    logger.info(f"Calculated Class Weights: {dict(zip(classes, np.round(class_weights_np, 3)))}")

    train_ds = KeypointSequenceDataset(X_train, y_train)
    val_ds = KeypointSequenceDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Instantiate Model
    N, T, V, C = X.shape
    model = DistressSpatioTemporalNet(
        num_keypoints=V,
        channels=C,
        num_classes=len(cfg.get("sequence", {}).get("labels", {0: "N", 1: "H", 2: "A", 3: "F"}))
    )
    logger.info(f"Model instantiated with {model.count_parameters():,} trainable parameters.")

    # Instantiate Trainer
    trainer = DistressModelTrainer(
        model=model,
        device=device,
        lr=args.lr,
        use_focal_loss=args.focal_loss,
        class_weights=class_weights_tensor
    )

    # Train
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        checkpoint_path=args.output
    )

    logger.info(f"=== TIER-2 TRAINING FINISHED (Best Macro F1: {results['best_macro_f1']:.4f}) ===")


if __name__ == "__main__":
    main()
