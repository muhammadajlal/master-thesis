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
    else:
        raise ValueError(
            f"Unknown connector_type={connector_type!r}. "
            f"Choose from: qformer, linear, mlp, pooling_mlp"
        )
