"""
Isolated In-House Spatio-Temporal Distress Model Architecture.
Target Tensor Shape: (batch_size, sequence_length=30, num_keypoints=17, channels=3) [x, y, confidence].
Features: Spatial Joint Embedding + 1D Conv + Bi-GRU + Temporal Self-Attention Pooling.
Parameter Count: ~950K (under 1.5M limit).
"""

import math
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSelfAttentionPooling(nn.Module):
    """Scaled Dot-Product Temporal Self-Attention Pooling layer."""

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch_Size, Temporal_Window, Embed_Dim)
        Returns:
            Context vector of shape (Batch_Size, Embed_Dim)
        """
        B, T, E = x.shape
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)  # (B, H, T, d)
        context = context.transpose(1, 2).contiguous().view(B, T, E)
        output = self.out_proj(context)

        # Average pool weighted by attention vector across time T
        temp_attn = attn_weights.mean(dim=1).mean(dim=1, keepdim=True)  # (B, 1, T)
        pooled = torch.bmm(temp_attn, output).squeeze(1)  # (B, E)
        return pooled


class InHouseDistressNet(nn.Module):
    """Proprietary In-House Spatio-Temporal Distress Classifier."""

    CLASS_MAP = {
        0: "Normal",
        1: "Universal Help Signal",
        2: "Aggression/Struggle",
        3: "Fall"
    }

    def __init__(
        self,
        sequence_length: int = 30,
        num_keypoints: int = 17,
        channels: int = 3,
        embed_dim: int = 128,
        rnn_hidden: int = 128,
        num_classes: int = 4,
        dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.num_keypoints = num_keypoints
        self.channels = channels
        self.num_classes = num_classes

        # 1. Spatial Joint Embedding
        in_features = num_keypoints * channels
        self.spatial_embedding = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU()
        )

        # 2. 1D Temporal Convolutional Block
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )

        # 3. Recurrent Sequence Encoder (Bidirectional GRU)
        self.bigru = nn.GRU(
            input_size=embed_dim,
            hidden_size=rnn_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        rnn_out_dim = rnn_hidden * 2

        # 4. Temporal Self-Attention Pooling
        self.attention_pooling = TemporalSelfAttentionPooling(embed_dim=rnn_out_dim, num_heads=4, dropout=dropout)

        # 5. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(rnn_out_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (batch_size, 30, num_keypoints, channels) or (batch_size, 30, num_keypoints * channels)
        Output logits: (batch_size, num_classes)
        """
        if x.dim() == 4:
            B, T, V, C = x.shape
            x_flat = x.view(B * T, V * C)
        else:
            B, T, VC = x.shape
            x_flat = x.view(B * T, VC)

        # Spatial Embedding
        feat = self.spatial_embedding(x_flat).view(B, T, -1)

        # 1D Temporal Conv (B, embed_dim, T)
        feat_t = feat.transpose(1, 2)
        conv_out = self.temporal_conv(feat_t)
        feat_t = (feat_t + conv_out).transpose(1, 2)  # Residual

        # Bi-GRU Temporal Encoder
        rnn_out, _ = self.bigru(feat_t)

        # Attention Pooling
        pooled = self.attention_pooling(rnn_out)

        # Logits
        logits = self.classifier(pooled)
        return logits

    @classmethod
    def from_pretrained(cls, weights_path: Union[str, Path], device: str = "cpu") -> "InHouseDistressNet":
        """Factory method loading pre-trained weights."""
        model = cls()
        path = Path(weights_path)
        if path.exists():
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        return model

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    dummy = torch.randn(8, 30, 17, 3)
    net = InHouseDistressNet(num_keypoints=17, channels=3)
    out = net(dummy)
    print(f"InHouseDistressNet Test | Input: {dummy.shape} | Logits: {out.shape} | Params: {net.count_parameters():,}")
