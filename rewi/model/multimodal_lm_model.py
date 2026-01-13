# rewi/model/multimodal_lm_model.py
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from rewi.model.projection_layer import ProjectionLayer
from rewi.model.pretrainedLM import PretrainedLMDecoder, LMConfig


class MultimodalLMModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        ratio_ds: int,
        d_cnn: int,
        lm_cfg: LMConfig,
        proj_dropout: float = 0.0,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.ratio_ds = int(max(1, ratio_ds))
        self.d_cnn = int(d_cnn)

        # Respect cfg flag for encoder freezing
        for p in self.encoder.parameters():
            p.requires_grad = (not freeze_encoder)

        # LM wrapper + projection
        self.lm = PretrainedLMDecoder(lm_cfg)
        self.proj = ProjectionLayer(
            d_in=self.d_cnn,
            d_out=self.lm.d_model,
            dropout=proj_dropout,
        )

        # Sanity checks (now work with your ProjectionLayer properties)
        assert self.proj.in_features == self.d_cnn, (
            f"Projection in_features={self.proj.in_features} != d_cnn={self.d_cnn}"
        )
        assert self.proj.out_features == self.lm.d_model, (
            f"Projection out_features={self.proj.out_features} != lm.d_model={self.lm.d_model}"
        )

        self._dbg_done = False

    def _call_encoder(self, x: torch.Tensor, len_x: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Support both encoder(x) and encoder(x, len_x).
        BLConv takes only x; other encoders may take (x, len_x).
        """
        try:
            return self.encoder(x, len_x)
        except TypeError:
            return self.encoder(x)

    def _to_BTC(self, enc_out: torch.Tensor) -> torch.Tensor:
        """
        Normalize encoder output layout to (B, T, C=d_cnn).
        Accepts either (B, T, d_cnn) or (B, d_cnn, T).
        """
        if enc_out.dim() != 3:
            raise ValueError(f"Expected encoder output 3D, got shape={tuple(enc_out.shape)}")

        # Already (B, T, C)
        if enc_out.shape[-1] == self.d_cnn:
            return enc_out

        # (B, C, T) -> (B, T, C)
        if enc_out.shape[1] == self.d_cnn:
            return enc_out.transpose(1, 2).contiguous()

        raise ValueError(
            f"Cannot infer encoder layout. enc_out.shape={tuple(enc_out.shape)} but d_cnn={self.d_cnn}. "
            f"Expected either (B,T,d_cnn) or (B,d_cnn,T)."
        )

    def _make_len_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """
        lengths: (B,)
        returns mask: (B, max_len) with 1=valid, 0=pad
        """
        device = lengths.device
        rng = torch.arange(max_len, device=device)[None, :]
        return (rng < lengths[:, None]).to(torch.long)

    def _encode(self, x: torch.Tensor, len_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            enc_states: (B, T, d_model)
            enc_mask:   (B, T) int mask (1=valid, 0=pad)
        """
        enc_out = self._call_encoder(x, len_x)
        enc_out = self._to_BTC(enc_out)  # (B, T, d_cnn)

        B, T, C = enc_out.shape

        # For BLConv-like stride-2 stacks, output length behaves like floor(len_x / ratio_ds).
        # This is safer than ceil; clamp to valid range.
        len_enc = torch.clamp(len_x // self.ratio_ds, min=1, max=T)

        enc_mask = self._make_len_mask(len_enc, T)  # (B, T)

        # Project to LM d_model
        enc_states = self.proj(enc_out)  # (B, T, d_model)

        if not self._dbg_done:
            self._dbg_done = True
            print(
                "[LMModel dbg] enc_out:", tuple(enc_out.shape),
                "| proj:", (self.d_cnn, "->", self.lm.d_model),
                "| ratio_ds:", self.ratio_ds
            )
            print(
                "[LMModel dbg] len_x min/med/max:",
                int(len_x.min()), int(len_x.median()), int(len_x.max())
            )
            print(
                "[LMModel dbg] len_enc min/med/max:",
                int(len_enc.min()), int(len_enc.median()), int(len_enc.max())
            )

        return enc_states, enc_mask

    def forward(self, x: torch.Tensor, len_x: torch.Tensor, labels: torch.Tensor):
        enc_states, enc_mask = self._encode(x, len_x)
        return self.lm(enc_states, enc_mask, labels)

    @torch.no_grad()
    def generate(self, x: torch.Tensor, len_x: torch.Tensor) -> List[str]:
        # Use the same encode path as training (robust to encoders with/without len_x)
        enc_states, enc_mask = self._encode(x, len_x)
        return self.lm.generate(enc_states, enc_mask)
