# rewi/model/qformer.py
"""
Q-Former Connector
==================

BLIP-2 style Q-Former that compresses variable-length encoder features
into a fixed number of "modality tokens" via learned queries cross-attending
to projected encoder states.

References:
    - Li et al., "BLIP-2", arXiv:2301.12597
    - Alayrac et al., "Flamingo", arXiv:2204.14198
"""
from __future__ import annotations

import torch
import torch.nn as nn


class QFormerConnector(nn.Module):
    """Compress encoder features (B, T, d_enc) → (B, K, d_lm) via cross-attention.

    Learned query vectors cross-attend to projected encoder states through a
    small Transformer decoder stack, producing a fixed-length set of
    "modality tokens" that can be prepended to an LM's input.

    Args:
        d_enc:        Encoder output dimension.
        d_lm:         LM hidden dimension (output dimension).
        num_queries:  Number of output modality tokens (K).
        num_layers:   Depth of the cross-attention Transformer.
        nhead:        Number of attention heads.
        dim_ff:       Feed-forward hidden size (default: 4 * d_lm).
        dropout:      Dropout rate inside Transformer layers.
    """

    def __init__(
        self,
        d_enc: int,
        d_lm: int,
        num_queries: int = 32,
        num_layers: int = 4,
        nhead: int = 8,
        dim_ff: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_enc = d_enc
        self.d_lm = d_lm
        self.num_queries = num_queries

        # Project encoder features to LM dimension
        self.proj_in = nn.Linear(d_enc, d_lm)
        self.proj_ln = nn.LayerNorm(d_lm)

        # Learned query vectors  (1, K, d_lm)  — expanded per batch in forward
        self.queries = nn.Parameter(torch.randn(1, num_queries, d_lm) * 0.02)

        # Transformer decoder: queries (tgt) cross-attend to encoder memory
        if dim_ff is None:
            dim_ff = d_lm * 4
        layer = nn.TransformerDecoderLayer(
            d_model=d_lm,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers)

        # Final layer norm on output tokens
        self.out_ln = nn.LayerNorm(d_lm)

    def forward(
        self,
        enc_states: torch.Tensor,
        enc_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            enc_states: (B, T, d_enc) encoder features.
            enc_mask:   (B, T) with 1 = valid, 0 = pad  (int or bool).

        Returns:
            (B, K, d_lm) compressed modality tokens.
        """
        B = enc_states.size(0)

        # Project encoder output to LM hidden size
        memory = self.proj_ln(self.proj_in(enc_states))  # (B, T, d_lm)

        # Expand learned queries for the batch
        queries = self.queries.expand(B, -1, -1)  # (B, K, d_lm)

        # Build key_padding_mask: True = ignore (PyTorch convention)
        memory_key_padding_mask = None
        if enc_mask is not None:
            # enc_mask uses 1=valid, 0=pad  →  True=pad for PyTorch
            memory_key_padding_mask = (enc_mask == 0)  # (B, T)

        # Cross-attend: queries attend to encoder memory
        out = self.transformer(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # (B, K, d_lm)

        return self.out_ln(out)


class MiniQFormerConnector(nn.Module):
    """Param-matched Q-Former: operates in a smaller internal dimension.

    Uses d_inner < d_lm for the cross-attention Transformer, with linear
    projections at input (d_enc → d_inner) and output (d_inner → d_lm).
    This keeps the cross-attention mechanism while matching MLP param budget.

    Args:
        d_enc:        Encoder output dimension.
        d_lm:         LM hidden dimension (final output dimension).
        d_inner:      Internal Transformer dimension (controls param count).
        num_queries:  Number of output modality tokens (K).
        num_layers:   Depth of the cross-attention Transformer.
        nhead:        Number of attention heads (must divide d_inner).
        dim_ff:       Feed-forward hidden size (default: 4 * d_inner).
        dropout:      Dropout rate inside Transformer layers.
    """

    def __init__(
        self,
        d_enc: int,
        d_lm: int,
        d_inner: int = 152,
        num_queries: int = 16,
        num_layers: int = 2,
        nhead: int = 4,
        dim_ff: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_enc = d_enc
        self.d_lm = d_lm
        self.d_inner = d_inner
        self.num_queries = num_queries

        # Project encoder features to internal dimension
        self.proj_in = nn.Linear(d_enc, d_inner)
        self.proj_in_ln = nn.LayerNorm(d_inner)

        # Learned query vectors in internal dimension
        self.queries = nn.Parameter(torch.randn(1, num_queries, d_inner) * 0.02)

        # Transformer decoder in internal dimension
        if dim_ff is None:
            dim_ff = d_inner * 4
        layer = nn.TransformerDecoderLayer(
            d_model=d_inner,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers)

        # Project back to LM dimension
        self.proj_out = nn.Linear(d_inner, d_lm)
        self.out_ln = nn.LayerNorm(d_lm)

    def forward(
        self,
        enc_states: torch.Tensor,
        enc_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            enc_states: (B, T, d_enc) encoder features.
            enc_mask:   (B, T) with 1 = valid, 0 = pad.

        Returns:
            (B, K, d_lm) compressed modality tokens.
        """
        B = enc_states.size(0)

        memory = self.proj_in_ln(self.proj_in(enc_states))  # (B, T, d_inner)
        queries = self.queries.expand(B, -1, -1)  # (B, K, d_inner)

        memory_key_padding_mask = None
        if enc_mask is not None:
            memory_key_padding_mask = (enc_mask == 0)

        out = self.transformer(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # (B, K, d_inner)

        return self.out_ln(self.proj_out(out))
