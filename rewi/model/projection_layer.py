# rewi/model/projection_layer.py
from __future__ import annotations
import torch
import torch.nn as nn

class ProjectionLayer(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)
        self.ln = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)

    @property
    def in_features(self) -> int:
        return self.proj.in_features

    @property
    def out_features(self) -> int:
        return self.proj.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_in)
        x = self.proj(x)
        x = self.ln(x)
        x = self.drop(x)
        return x
