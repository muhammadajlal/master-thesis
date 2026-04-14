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


class CTCPosteriorProjector(nn.Module):
    """K2: LegoSLM-style CTC posterior embedding reconstruction.

    Reconstructs connector output as weighted sum of LM vocabulary embeddings
    using CTC posteriors:

        e_t = Σ_c P(c|frame_t) · LM_embed(c)

    The modality gap is eliminated by construction since the output lives
    in the LM's native embedding space. Requires a mapping from CTC
    vocabulary to LM token IDs.

    This connector is special: it needs the CTC logits and the LM's
    embedding matrix at forward time. The VLM model must pass these.

    Param count: ~0 extra (reuses CTC head + LM embeddings).
    """

    def __init__(self, d_enc: int, d_lm: int, num_tokens: int = 16):
        super().__init__()
        self.d_enc = d_enc
        self.d_lm = d_lm
        self.num_tokens = num_tokens
        self.pool = nn.AdaptiveAvgPool1d(num_tokens)
        # Lightweight residual MLP for refinement
        # Initialized near-zero so output starts as pure CTC posterior reconstruction
        self.residual_mlp = nn.Sequential(
            nn.Linear(d_lm, d_lm),
            nn.GELU(),
            nn.Linear(d_lm, d_lm),
        )
        # Init last linear near-zero for residual start
        nn.init.zeros_(self.residual_mlp[2].weight)
        nn.init.zeros_(self.residual_mlp[2].bias)
        # CTC vocab → LM token ID mapping (set by VLM model during init)
        self.register_buffer("ctc_to_lm_ids", None)

    def set_ctc_to_lm_mapping(self, ctc_to_lm_ids: torch.Tensor):
        """Set CTC vocab → LM token ID mapping.

        Args:
            ctc_to_lm_ids: (V_ctc,) LM token IDs for each CTC class.
        """
        self.register_buffer("ctc_to_lm_ids", ctc_to_lm_ids)

    def forward(
        self,
        enc_states: torch.Tensor,
        enc_mask: torch.Tensor | None = None,
        ctc_logits: torch.Tensor | None = None,
        lm_embed_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with CTC posterior reconstruction.

        If ctc_logits and lm_embed_weight are provided, reconstructs
        embeddings from CTC posteriors. Otherwise falls back to standard
        pooling + projection.

        Args:
            enc_states: (B, T, d_enc)
            enc_mask: (B, T) optional mask
            ctc_logits: (B, T, V_ctc) CTC logits from hybrid head
            lm_embed_weight: (V_lm, d_lm) LM embedding matrix

        Returns:
            (B, K, d_lm) reconstructed embeddings
        """
        if ctc_logits is not None and lm_embed_weight is not None and self.ctc_to_lm_ids is not None:
            return self._forward_ctc_posterior(
                enc_states, enc_mask, ctc_logits, lm_embed_weight,
            )

        # Fallback: standard pooling + linear (for generation/eval)
        x = enc_states.transpose(1, 2)  # (B, C, T)
        if enc_mask is not None:
            x = x * enc_mask.unsqueeze(1).float()
        x = self.pool(x).transpose(1, 2)  # (B, K, C)
        # Can't project without LM embeddings, so just return zeros of right shape
        B = x.size(0)
        return torch.zeros(B, self.num_tokens, self.d_lm, device=x.device)

    def _forward_ctc_posterior(
        self,
        enc_states: torch.Tensor,
        enc_mask: torch.Tensor | None,
        ctc_logits: torch.Tensor,
        lm_embed_weight: torch.Tensor,
    ) -> torch.Tensor:
        B, T, V_ctc = ctc_logits.shape

        # CTC posteriors (softmax over vocab)
        ctc_probs = torch.softmax(ctc_logits, dim=-1)  # (B, T, V_ctc)

        # Pool CTC probs to K tokens
        # (B, T, V_ctc) → (B, V_ctc, T) → pool → (B, V_ctc, K) → (B, K, V_ctc)
        probs_pooled = self.pool(ctc_probs.transpose(1, 2)).transpose(1, 2)  # (B, K, V_ctc)

        # Re-normalize after pooling
        probs_pooled = probs_pooled / probs_pooled.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Gather LM embeddings for CTC vocab: (V_ctc, d_lm)
        ctc_lm_emb = lm_embed_weight[self.ctc_to_lm_ids]  # (V_ctc, d_lm)

        # Weighted sum: (B, K, V_ctc) @ (V_ctc, d_lm) → (B, K, d_lm)
        reconstructed = torch.matmul(probs_pooled, ctc_lm_emb)

        # Add residual refinement
        output = reconstructed + self.residual_mlp(reconstructed)

        return output


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
    elif ctype in ("ctc_posterior", "lego"):
        k = pool_tokens if pool_tokens is not None else num_queries
        return CTCPosteriorProjector(d_enc, d_lm, num_tokens=k)
    elif ctype in ("kv_multiview", "kv"):
        k = pool_tokens if pool_tokens is not None else num_queries
        from rewi.model.kv_injection import KVInjectionWrapper
        return KVInjectionWrapper(d_enc, d_lm, num_tokens=k, num_views=4)
    elif ctype in ("kv_slim", "kv_multiview_slim"):
        k = pool_tokens if pool_tokens is not None else num_queries
        from rewi.model.kv_injection import KVInjectionSlim
        return KVInjectionSlim(d_enc, d_lm, num_tokens=k, num_views=2, d_hidden=384)
    elif ctype in ("mini_qformer", "qformer_mini"):
        from rewi.model.qformer import MiniQFormerConnector
        return MiniQFormerConnector(
            d_enc=d_enc,
            d_lm=d_lm,
            d_inner=152,
            num_queries=num_queries,
            num_layers=num_layers,
            nhead=4,
            dropout=dropout,
        )
    else:
        raise ValueError(
            f"Unknown connector_type={connector_type!r}. "
            f"Choose from: qformer, mini_qformer, linear, mlp, pooling_mlp, "
            f"conv_pool, ctc_posterior, kv_multiview, kv_slim"
        )
