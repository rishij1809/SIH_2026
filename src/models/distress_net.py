"""
Tier-2 In-House Spatio-Temporal Model Architecture for Distress Gesture Recognition.
Combines 1D Spatial Feature Embedding, Bidirectional GRU/LSTM, and Temporal Self-Attention.
Engineered for ultra-low latency on low-resource edge CCTV deployment.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSelfAttention(nn.Module):
    """Scaled Dot-Product Temporal Self-Attention module."""

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (Batch_Size, Temporal_Window, Embed_Dim)

        Returns:
            Tuple of (context_vector (B, Embed_Dim), attention_weights (B, Heads, T, T))
        """
        B, T, E = x.shape
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, d)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, d)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, d)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, T, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, V)  # (B, H, T, d)
        context = context.transpose(1, 2).contiguous().view(B, T, E)  # (B, T, E)
        output = self.out_proj(context)

        # Average pool across temporal sequence weighted by attention score
        temp_attn = attn_weights.mean(dim=1).mean(dim=1, keepdim=True)  # (B, 1, T)
        pooled_context = torch.bmm(temp_attn, output).squeeze(1)  # (B, E)

        return pooled_context, attn_weights


class SpatialJointEmbedding(nn.Module):
    """Extracts spatial joint relationship features from skeletal keypoints."""

    def __init__(self, num_keypoints: int = 33, channels: int = 3, embed_dim: int = 128) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.channels = channels
        in_dim = num_keypoints * channels

        self.spatial_fc = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, V, C) or (B, T, V*C)

        Returns:
            Embedded features of shape (B, T, embed_dim)
        """
        if x.dim() == 4:
            B, T, V, C = x.shape
            x = x.contiguous().view(B * T, V * C)
        else:
            B, T, VC = x.shape
            x = x.contiguous().view(B * T, VC)

        embedded = self.spatial_fc(x)
        return embedded.view(B, T, -1)


class DistressSpatioTemporalNet(nn.Module):
    """
    Tier-2 Lightweight Spatio-Temporal AI Model for Real-Time Safety Surveillance.
    Combines Spatial Joint Embedding + 1D Conv Temporal Encoder + Bi-GRU + Temporal Self-Attention.
    """

    def __init__(
        self,
        num_keypoints: int = 33,
        channels: int = 3,
        spatial_dim: int = 128,
        rnn_hidden_dim: int = 128,
        num_rnn_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.channels = channels
        self.num_classes = num_classes

        # 1. Spatial Embedding
        self.spatial_embed = SpatialJointEmbedding(
            num_keypoints=num_keypoints,
            channels=channels,
            embed_dim=spatial_dim
        )

        # 2. 1D Temporal Convolutional Block
        self.conv1d_temporal = nn.Sequential(
            nn.Conv1d(spatial_dim, spatial_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(spatial_dim),
            nn.ReLU(),
            nn.Conv1d(spatial_dim, spatial_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(spatial_dim),
            nn.ReLU()
        )

        # 3. Recurrent Sequence Encoder (Bidirectional GRU)
        self.bigru = nn.GRU(
            input_size=spatial_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_rnn_layers > 1 else 0.0
        )

        rnn_out_dim = rnn_hidden_dim * 2

        # 4. Temporal Self-Attention
        self.attention = TemporalSelfAttention(embed_dim=rnn_out_dim, num_heads=4, dropout=dropout)

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
        Forward pass.
        Input x: (Batch_Size, Temporal_Window, Num_Keypoints, Channels)
        Output logits: (Batch_Size, Num_Classes)
        """
        # Step 1: Spatial Joint Embedding (B, T, D_spatial)
        feat = self.spatial_embed(x)

        # Step 2: 1D Temporal Convolution (B, D_spatial, T)
        feat_t = feat.transpose(1, 2)
        conv_feat = self.conv1d_temporal(feat_t)
        feat_t = (feat_t + conv_feat).transpose(1, 2)  # Residual connection

        # Step 3: Bi-GRU Temporal Feature Learning (B, T, 2*rnn_hidden_dim)
        rnn_out, _ = self.bigru(feat_t)

        # Step 4: Temporal Self-Attention Aggregation (B, 2*rnn_hidden_dim)
        context_vector, _ = self.attention(rnn_out)

        # Step 5: Classification Logits (B, num_classes)
        logits = self.classifier(context_vector)
        return logits

    def count_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
