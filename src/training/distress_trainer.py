"""
Tier-2 PyTorch Training & Evaluation Engine for Spatio-Temporal Distress Model.
Supports Mixed Precision AMP, Focal Loss, Cosine Annealing, Macro F1 checkpointing, and TensorBoard.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except Exception:
    HAS_TENSORBOARD = False
    class SummaryWriter:
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def close(self): pass

from src.models.distress_net import DistressSpatioTemporalNet
from src.training.focal_loss import FocalLoss
from src.utils.logger import setup_logger

logger = setup_logger("DistressTrainer")


class KeypointSequenceDataset(Dataset):
    """PyTorch Dataset for (N, T, V, C) keypoint sequence matrices."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Args:
            X: Array of shape (N, T, V, C)
            y: Array of shape (N,)
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class DistressModelTrainer:
    """Trainer engine for Tier-2 Spatio-Temporal Distress Gesture Recognizer."""

    def __init__(
        self,
        model: DistressSpatioTemporalNet,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        use_focal_loss: bool = True,
        class_weights: Optional[torch.Tensor] = None,
        log_dir: str = "runs/distress_experiment"
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.use_amp = (self.device.type == "cuda")

        # Loss selection
        if use_focal_loss:
            self.criterion = FocalLoss(gamma=2.0, alpha=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-5
        )

        # PyTorch AMP Scaler
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        
        # TensorBoard Logger
        self.writer = SummaryWriter(log_dir=log_dir)

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Runs single training epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_b, y_b in dataloader:
            X_b, y_b = X_b.to(self.device), y_b.to(self.device)
            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(X_b)
                loss = self.criterion(logits, y_b)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * X_b.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_b).sum().item()
            total += X_b.size(0)

        epoch_loss = total_loss / max(1, total)
        epoch_acc = correct / max(1, total)
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluates model performance and returns macro F1, precision, recall, accuracy."""
        self.model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0.0
        total = 0

        for X_b, y_b in dataloader:
            X_b, y_b = X_b.to(self.device), y_b.to(self.device)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(X_b)
                loss = self.criterion(logits, y_b)

            total_loss += loss.item() * X_b.size(0)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_b.cpu().numpy())
            total += X_b.size(0)

        val_loss = total_loss / max(1, total)
        acc = accuracy_score(all_targets, all_preds)
        macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        precision = precision_score(all_targets, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)
        cm = confusion_matrix(all_targets, all_preds)

        return {
            "loss": val_loss,
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "precision": float(precision),
            "recall": float(recall),
            "confusion_matrix": cm
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 40,
        checkpoint_path: str = "models/distress_model.pt"
    ) -> Dict[str, float]:
        """Full training loop with Macro F1 checkpoint saving and TensorBoard logging."""
        best_macro_f1 = -1.0
        ckpt_path = Path(checkpoint_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting Tier-2 Training for {epochs} epochs on device: {self.device}...")

        pbar = tqdm(range(1, epochs + 1), desc="Tier-2 Training")
        for epoch in pbar:
            train_loss, train_acc = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # TensorBoard logging
            self.writer.add_scalar("Train/Loss", train_loss, epoch)
            self.writer.add_scalar("Train/Accuracy", train_acc, epoch)
            self.writer.add_scalar("Val/Loss", val_metrics["loss"], epoch)
            self.writer.add_scalar("Val/Macro_F1", val_metrics["macro_f1"], epoch)
            self.writer.add_scalar("Val/Accuracy", val_metrics["accuracy"], epoch)
            self.writer.add_scalar("LearningRate", current_lr, epoch)

            pbar.set_postfix({
                "Tr_Loss": f"{train_loss:.3f}",
                "Val_F1": f"{val_metrics['macro_f1']:.3f}",
                "Val_Acc": f"{val_metrics['accuracy']:.3f}"
            })

            # Checkpoint saving on peak Macro F1-score
            if val_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = val_metrics["macro_f1"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "macro_f1": best_macro_f1,
                        "val_metrics": val_metrics
                    },
                    ckpt_path
                )

        logger.info(f"Training Complete! Peak Validation Macro F1-Score: {best_macro_f1:.4f}")
        logger.info(f"Saved best model checkpoint -> {ckpt_path}")
        self.writer.close()
        return {"best_macro_f1": best_macro_f1}
