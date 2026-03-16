# rewi/model/multimodal_lm_model.py
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from rewi.model.projection_layer import ProjectionLayer
from rewi.model.pretrainedLM import PretrainedLMDecoder, LMConfig
from rewi.model.prompt import PromptManager


class MultimodalLMModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        ratio_ds: int,
        d_cnn: int,
        lm_cfg: LMConfig,
        proj_dropout: float = 0.0,
        freeze_encoder: bool = False,
        vocab_ctc: int | None = None,
        num_soft_tokens: int = 0,
        prompt_text: str = "",
    ):
        super().__init__()
        self.encoder = encoder
        self.ratio_ds = int(max(1, ratio_ds))
        self.d_cnn = int(d_cnn)
        self.hybrid_mode = vocab_ctc is not None

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

        # Optional CTC auxiliary head on raw encoder features
        self.ctc_head: nn.Module | None = None
        if vocab_ctc is not None:
            self.ctc_head = nn.Linear(self.d_cnn, vocab_ctc)

        # Task prefix: soft learned tokens + optional fixed text prompt
        self.prompt: PromptManager | None = None
        if num_soft_tokens > 0 or prompt_text:
            self.prompt = PromptManager(
                d_lm=self.lm.d_model,
                prompt_text=prompt_text,
                num_soft_tokens=num_soft_tokens,
                tokenizer=self.lm.tokenizer,
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

    def _encode(self, x: torch.Tensor, len_x: torch.Tensor):
        """
        Returns:
            enc_states: (B, T', d_model) — projected features for LM (T' = P + T if prompt)
            enc_mask:   (B, T') int mask (1=valid, 0=pad)
            enc_out:    (B, T, d_cnn)    — raw encoder features (for CTC head, no prefix)
            len_enc:    (B,)             — encoder output lengths (no prefix)
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

        # Prepend task prefix: [soft_prefix | fixed_prompt | projected_features]
        if self.prompt is not None and self.prompt.total_prefix_len > 0:
            embed_fn = self.lm.shared_embedding  # T5's input embedding
            prefix = self.prompt.get_prefix_embeds(embed_fn, B, enc_states.device)
            enc_states = torch.cat([prefix, enc_states], dim=1)  # (B, P+T, d_model)
            P = prefix.size(1)
            prefix_mask = torch.ones(B, P, device=enc_mask.device, dtype=enc_mask.dtype)
            enc_mask = torch.cat([prefix_mask, enc_mask], dim=1)  # (B, P+T)

        if not self._dbg_done:
            self._dbg_done = True
            prompt_info = f"| prompt: {self.prompt.total_prefix_len} tokens" if self.prompt else ""
            print(
                "[LMModel dbg] enc_out:", tuple(enc_out.shape),
                "| proj:", (self.d_cnn, "->", self.lm.d_model),
                "| ratio_ds:", self.ratio_ds,
                "| hybrid:", self.hybrid_mode,
                prompt_info
            )
            print(
                "[LMModel dbg] len_x min/med/max:",
                int(len_x.min()), int(len_x.median()), int(len_x.max())
            )
            print(
                "[LMModel dbg] len_enc min/med/max:",
                int(len_enc.min()), int(len_enc.median()), int(len_enc.max())
            )
            print(
                "[LMModel dbg] enc_states (after prefix):", tuple(enc_states.shape),
                "| enc_mask:", tuple(enc_mask.shape)
            )

        return enc_states, enc_mask, enc_out, len_enc

    def forward(self, x: torch.Tensor, len_x: torch.Tensor, labels: torch.Tensor):
        enc_states, enc_mask, enc_out, len_enc = self._encode(x, len_x)
        lm_out = self.lm(enc_states, enc_mask, labels)

        if not self.hybrid_mode:
            return lm_out

        result = {"lm_out": lm_out, "enc_lengths": len_enc}
        if self.ctc_head is not None:
            result["ctc_logits"] = self.ctc_head(enc_out)  # (B, T, vocab_ctc)
        return result

    @torch.no_grad()
    def generate(self, x: torch.Tensor, len_x: torch.Tensor) -> List[str]:
        # Use the same encode path as training (robust to encoders with/without len_x)
        enc_states, enc_mask, _, _ = self._encode(x, len_x)
        return self.lm.generate(enc_states, enc_mask)
