"""
Conformer Encoder for IMU Handwriting Recognition
==================================================

Self-contained implementation (no torchaudio dependency) of the
Conformer architecture (Gulati et al., 2020):

    FFN(½) → MHSA → Conv → FFN(½) → LayerNorm

Each block combines multi-head self-attention with depthwise
convolution, capturing both global and local temporal patterns.

Default config ``conformer_b`` targets ~2.56M params (matching
BLConv's ~2.46M) with d_model=256, d_ff=640, 2 layers, subsample=8.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvSubsampler(nn.Module):
    """Convolutional front-end for temporal downsampling.

    Two strided Conv1d layers reduce the temporal dimension by a
    configurable factor (default 8x to match BLConv).

    Input:  (B, C_in, T)
    Output: (B, T', d_model)
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        subsample: int = 8,
    ):
        super().__init__()
        if subsample == 8:
            layers = [
                nn.Conv1d(in_channels, d_model // 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(d_model // 2),
                nn.GELU(),
                nn.Conv1d(d_model // 2, d_model, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
            ]
        elif subsample == 4:
            layers = [
                nn.Conv1d(in_channels, d_model // 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(d_model // 2),
                nn.GELU(),
                nn.Conv1d(d_model // 2, d_model, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
            ]
        else:
            raise ValueError(f"subsample must be 4 or 8, got {subsample}")

        self.net = nn.Sequential(*layers)
        self.subsample = subsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C_in, T) → (B, T', d_model)"""
        x = self.net(x)          # (B, d_model, T')
        return x.transpose(1, 2)  # (B, T', d_model)


class FeedForwardModule(nn.Module):
    """Conformer feed-forward module: LN → Linear → Swish → Drop → Linear → Drop."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        x = F.silu(self.w1(x))   # Swish activation
        x = self.drop(x)
        x = self.w2(x)
        x = self.drop(x)
        return x


class ConvolutionModule(nn.Module):
    """Conformer convolution module.

    LN → Pointwise(d→2d) → GLU → DepthwiseConv → BN → Swish → Pointwise(d→d) → Drop
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=d_model, bias=False,
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.pointwise2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) → (B, T, D)"""
        x = self.ln(x)
        x = x.transpose(1, 2)       # (B, D, T) for Conv1d
        x = self.pointwise1(x)      # (B, 2D, T)
        x = F.glu(x, dim=1)         # (B, D, T)
        x = self.depthwise(x)       # (B, D, T)
        x = self.bn(x)
        x = F.silu(x)               # Swish
        x = self.pointwise2(x)      # (B, D, T)
        x = self.drop(x)
        return x.transpose(1, 2)    # (B, T, D)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with relative positional encoding (simplified).

    Uses standard nn.MultiheadAttention with LayerNorm pre-processing.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """(B, T, D) → (B, T, D)"""
        x = self.ln(x)
        out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        return self.drop(out)


class ConformerBlock(nn.Module):
    """Single Conformer block: FFN(½) → MHSA → Conv → FFN(½) → LN."""

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        d_ff: int = 1024,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, d_ff, dropout)
        self.mhsa = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ff2 = FeedForwardModule(d_model, d_ff, dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.mhsa(x, key_padding_mask=key_padding_mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.ln(x)


class ConformerEncoder(nn.Module):
    """Conformer encoder for IMU time-series.

    Input:  (B, C, T)  — raw IMU channels
    Output: (B, T', d_model)  — encoded features

    Attributes:
        dim_out:  Output feature dimension (= d_model).
        ratio_ds: Temporal downsampling ratio.
    """

    def __init__(
        self,
        in_channels: int = 13,
        d_model: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        d_ff: int = 640,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        subsample: int = 8,
    ):
        super().__init__()
        self.frontend = ConvSubsampler(in_channels, d_model, subsample)

        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, num_heads, d_ff, conv_kernel_size, dropout)
            for _ in range(num_layers)
        ])

        self.dim_out = d_model
        self.ratio_ds = subsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) → (B, T', d_model)"""
        x = self.frontend(x)  # (B, T', d_model)
        for block in self.blocks:
            x = block(x)
        return x
