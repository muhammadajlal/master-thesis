# rewi/model/projectors.py
"""
Lightweight Modality Projectors
===============================

Simple connectors that map encoder features ``(B, T, d_enc)`` to LM
embedding space ``(B, ?, d_lm)``.  These are deliberately lightweight
alternatives to the Q-Former, designed to test whether the **pretrained
LM already provides the linguistic knowledge** and a simple bridge suffices.

Inspired by:
  - LLaVA (Liu et al., 2023):  single ``Linear``
  - LLaVA-1.5 (Liu et al., 2024):  2-layer MLP with GELU  — "the FC
    connector is surprisingly powerful and data-efficient"
  - Perna et al. (Interspeech 2025):  Base/Conv adapters for speech→LLM

All projectors expose the same ``forward(enc_states, enc_mask) → Tensor``
interface as :class:`QFormerConnector`, so ``VLMModel`` can swap them
transparently.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LinearProjector(nn.Module):
    """Simple linear projection + LayerNorm  (LLaVA-v1 style).

    Maps each encoder frame independently:  ``d_enc → d_lm``.
    Output length equals encoder length (no compression).

    Param count:  ``d_enc × d_lm + d_lm  +  2 × d_lm``  (linear + LN)
                  e.g.  512 × 768 + 768 + 2×768  ≈  **395 K**
    """

    def __init__(self, d_enc: int, d_lm: int):
        super().__init__()
        self.proj = nn.Linear(d_enc, d_lm)
        self.ln = nn.LayerNorm(d_lm)

    def forward(
        self,
        enc_states: torch.Tensor,
        enc_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``(B, T, d_enc)`` → ``(B, T, d_lm)``."""
        return self.ln(self.proj(enc_states))


class MLPProjector(nn.Module):
    """Two-layer MLP projection  (LLaVA-1.5 style).

    ``Linear(d_enc, d_lm) → GELU → Linear(d_lm, d_lm) → LayerNorm``.

    Output length equals encoder length (no compression).

    Param count:  ``d_enc×d_lm + d_lm  +  d_lm×d_lm + d_lm  +  2×d_lm``
                  e.g.  512→768:  393K + 591K + 1.5K  ≈  **985 K**
    """

    def __init__(self, d_enc: int, d_lm: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_enc, d_lm),
            nn.GELU(),
            nn.Linear(d_lm, d_lm),
            nn.LayerNorm(d_lm),
        )

    def forward(
        self,
        enc_states: torch.Tensor,
        enc_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``(B, T, d_enc)`` → ``(B, T, d_lm)``."""
        return self.net(enc_states)


class PoolingMLPProjector(nn.Module):
    """Adaptive temporal pooling followed by MLP projection.

    Reduces variable-length encoder output to a fixed number of tokens (K)
    before projection — provides a lightweight alternative to Q-Former's
    cross-attention compression.

    ``AdaptiveAvgPool1d(K) → Linear(d_enc, d_lm) → GELU
      → Linear(d_lm, d_lm) → LayerNorm``

    Param count:  same as MLPProjector (pooling is parameter-free).
    """

    def __init__(self, d_enc: int, d_lm: int, num_tokens: int = 16):
        super().__init__()
        self.num_tokens = num_tokens
        self.pool = nn.AdaptiveAvgPool1d(num_tokens)
        self.net = nn.Sequential(
            nn.Linear(d_enc, d_lm),
            nn.GELU(),
            nn.Linear(d_lm, d_lm),
            nn.LayerNorm(d_lm),
        )

    def forward(
        self,
        enc_states: torch.Tensor,
        enc_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``(B, T, d_enc)`` → ``(B, K, d_lm)``  where K = num_tokens."""
        # enc_states: (B, T, C) → pool needs (B, C, T)
        x = enc_states.transpose(1, 2)  # (B, C, T)

        # Mask padded positions before pooling (zero them out)
        if enc_mask is not None:
            x = x * enc_mask.unsqueeze(1).float()  # (B, C, T)

        x = self.pool(x)  # (B, C, K)
        x = x.transpose(1, 2)  # (B, K, C)
        return self.net(x)


class ConvPoolProjector(nn.Module):
    """1D Conv downsampling + linear projection — minimal connector.

    Uses a strided 1D convolution to temporally compress encoder output,
    then a single linear layer to project to LM dimension. Much lighter
    than Q-Former or MLP, tests whether the pretrained LM can work with
    minimal bridging.

    ``Conv1d(d_enc, d_enc, k, stride=s) → GELU → Linear(d_enc, d_lm) → LN``

    Param count:  ``d_enc × k × 1 + d_enc  +  d_enc × d_lm + d_lm  +  2 × d_lm``
                  e.g.  512, k=5, s=4, d_lm=768:
                        512×5 + 512 + 512×768 + 768 + 2×768 ≈ **397 K**
    """

    def __init__(
        self, d_enc: int, d_lm: int, kernel_size: int = 5, stride: int = 4,
    ):
        super().__init__()
        self.stride = stride
        # Depthwise-style: groups=1 keeps it simple, low param
        self.conv = nn.Conv1d(
            d_enc, d_enc, kernel_size=kernel_size, stride=stride,
            padding=kernel_size // 2, groups=1,
        )
        self.act = nn.GELU()
        self.proj = nn.Linear(d_enc, d_lm)
        self.ln = nn.LayerNorm(d_lm)

    def forward(
        self,
        enc_states: torch.Tensor,
        enc_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``(B, T, d_enc)`` → ``(B, T//stride, d_lm)``."""
        # (B, T, C) → (B, C, T) for Conv1d
        x = enc_states.transpose(1, 2)
        if enc_mask is not None:
            x = x * enc_mask.unsqueeze(1).float()
        x = self.act(self.conv(x))  # (B, C, T//s)
        x = x.transpose(1, 2)  # (B, T//s, C)
        return self.ln(self.proj(x))


def build_connector(
    connector_type: str,
    d_enc: int,
    d_lm: int,
    *,
    # Q-Former specific
    num_queries: int = 16,
    num_layers: int = 2,
    nhead: int = 8,
    dropout: float = 0.1,
    # Pooling specific
    pool_tokens: int | None = None,
) -> nn.Module:
    """Factory function to build a connector by name.

    Args:
        connector_type: One of ``"qformer"``, ``"linear"``, ``"mlp"``,
                        ``"pooling_mlp"``.
        d_enc:          Encoder output dimension.
        d_lm:           LM hidden dimension.

    Keyword Args:
        num_queries:    Q-Former learned queries (K).
        num_layers:     Q-Former depth.
        nhead:          Q-Former attention heads.
        dropout:        Q-Former dropout.
        pool_tokens:    For ``pooling_mlp``, number of output tokens.
                        Defaults to ``num_queries``.

    Returns:
        Connector module with ``forward(enc_states, enc_mask) → Tensor``.
    """
    ctype = connector_type.lower().replace("-", "_")

    if ctype == "qformer":
        from rewi.model.qformer import QFormerConnector
        return QFormerConnector(
            d_enc=d_enc,
            d_lm=d_lm,
            num_queries=num_queries,
            num_layers=num_layers,
            nhead=nhead,
            dropout=dropout,
        )
    elif ctype == "linear":
        return LinearProjector(d_enc, d_lm)
    elif ctype == "mlp":
        return MLPProjector(d_enc, d_lm)
    elif ctype in ("pooling_mlp", "pooling"):
        k = pool_tokens if pool_tokens is not None else num_queries
        return PoolingMLPProjector(d_enc, d_lm, num_tokens=k)
    elif ctype in ("conv_pool", "conv"):
        return ConvPoolProjector(d_enc, d_lm, kernel_size=5, stride=4)
    else:
        raise ValueError(
            f"Unknown connector_type={connector_type!r}. "
            f"Choose from: qformer, linear, mlp, pooling_mlp, conv_pool"
        )
