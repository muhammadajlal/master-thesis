import torch
import torch.nn as nn
import torch.nn.functional as F

from .ARDecoder import ARDecoder
from .builders import build_decoder, build_encoder


class BaseModel(nn.Module):
    """
    Supports both:
      - CTC pipeline (encoder + per-timestep decoder)
      - AR pipeline (encoder + ARDecoder with cross-attention)

    Optional decoder-side CTC: encoder frames attend to intermediate decoder
    layer states, then apply CTC loss using the AR vocab projection prefix.
    This places CTC regularization "between/with the decoder layers" per
    the professor's suggestion.
    """

    def __init__(
        self,
        arch_en: str,
        arch_de: str,
        in_chan: int,
        num_cls: int,
        len_seq: int = 0,
        *,
        use_gated_attention: bool = False,
        gating_type: str = "elementwise",
        vocab_ctc: int | None = None,
        pad_id: int | None = None,
        decoder_side_ctc_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        self.arch_en = arch_en
        self.arch_de = arch_de
        self.in_chan = in_chan
        self.num_cls = num_cls
        self.len_seq = len_seq
        self.vocab_ctc = vocab_ctc if vocab_ctc is not None else num_cls
        self.pad_id = pad_id

        decoder_side_ctc_cfg = decoder_side_ctc_cfg or {}
        self.decoder_side_ctc_enabled = bool(decoder_side_ctc_cfg.get("enabled", False))
        self.decoder_side_ctc_cfg = decoder_side_ctc_cfg
        self.decoder_side_ctc_mode = str(decoder_side_ctc_cfg.get("mode", "attention")).lower()  # "attention" or "pool"

        self.encoder = build_encoder(in_chan, arch_en, len_seq)
        self.decoder = build_decoder(
            self.encoder.dim_out,
            num_cls,
            arch_de,
            len_seq // self.encoder.ratio_ds if arch_en != 'trans' else 0,
            use_gated_attention=use_gated_attention,
            gating_type=gating_type,
        )

        # If AR decoder d_model != encoder dim, add a projection
        self.mem_proj = None
        if isinstance(self.decoder, ARDecoder):
            dec_dim = self.decoder.d_model
            enc_dim = self.encoder.dim_out
            if enc_dim != dec_dim:
                self.mem_proj = nn.Linear(enc_dim, dec_dim)

        # Decoder-side CTC: reverse cross-attention (encoder frames query decoder states)
        # or global pooling mode (no extra params)
        self.dec_ctc_attn = None
        self.dec_ctc_attn_ln = None
        if self.decoder_side_ctc_enabled and isinstance(self.decoder, ARDecoder):
            if self.decoder_side_ctc_mode == "attention":
                dec_dim = self.decoder.d_model
                nhead = int(decoder_side_ctc_cfg.get("nhead", getattr(self.decoder, "nhead", 4)))
                if dec_dim % nhead != 0:
                    raise ValueError(
                        f"decoder_side_ctc.nhead must divide decoder dim. Got d_dec={dec_dim}, nhead={nhead}"
                    )
                self.dec_ctc_attn = nn.MultiheadAttention(embed_dim=dec_dim, num_heads=nhead, batch_first=True)
                if bool(decoder_side_ctc_cfg.get("layernorm", True)):
                    self.dec_ctc_attn_ln = nn.LayerNorm(dec_dim)
            elif self.decoder_side_ctc_mode != "pool":
                raise ValueError(f"decoder_side_ctc.mode must be 'attention' or 'pool', got: {self.decoder_side_ctc_mode}")

    def _encode_with_mask(self, x: torch.Tensor, in_lengths: torch.Tensor | None):
        # infer raw lengths if not provided (before encoder)
        if in_lengths is None:
            with torch.no_grad():
                valid = (x.abs().sum(dim=1) > 1e-6)  # (B, T)
                in_lengths = valid.sum(dim=1)  # (B,)
        else:
            in_lengths = in_lengths.to(device=x.device)

        feats = self.encoder(x)  # (B, Tm, Cenc)
        Tm = feats.size(1)

        enc_lengths = torch.div(in_lengths, self.encoder.ratio_ds, rounding_mode='floor')
        enc_lengths = enc_lengths.clamp(min=1, max=Tm)
        enc_lengths = enc_lengths.to(device=feats.device, dtype=torch.long)

        enc_pad = torch.arange(Tm, device=feats.device).unsqueeze(0) >= enc_lengths.unsqueeze(1)
        return feats, enc_pad

    def forward(
        self,
        x: torch.Tensor,
        in_lengths: torch.Tensor | None = None,
        y_inp: torch.Tensor | None = None,
        *,
        return_ar_layers: bool = False,
        return_dec_ctc: bool = False,
    ):
        # AR path (teacher forcing or dummy path for profiling)
        if isinstance(self.decoder, ARDecoder):
            feats, enc_pad = self._encode_with_mask(x, in_lengths)
            mem = feats
            if self.mem_proj is not None:
                mem = self.mem_proj(feats)

            if y_inp is None:
                B = mem.size(0)
                y_inp = torch.zeros(B, 1, dtype=torch.long, device=mem.device)

            need_layers = return_ar_layers or (return_dec_ctc and self.decoder_side_ctc_enabled)

            if not need_layers:
                return self.decoder(y_inp, mem, enc_pad)  # (B, N, V)

            dec_out = self.decoder(y_inp, mem, enc_pad, return_layer_states=True)
            layer_states = dec_out["layer_states"]
            result = {
                "logits": dec_out["logits"],
                "layer_states": layer_states,
                "logits_layers": [self.decoder.proj(s) for s in layer_states],
            }

            # Decoder-side CTC: encoder frames attend to decoder layer states
            if return_dec_ctc and self.decoder_side_ctc_enabled:
                layers_cfg = self.decoder_side_ctc_cfg.get("layers", [1, 2, 3])
                if not isinstance(layers_cfg, (list, tuple)):
                    layers_cfg = [layers_cfg]
                layer_idxs = [int(l) - 1 for l in layers_cfg]
                layer_idxs = [i for i in layer_idxs if 0 <= i < len(layer_states)]
                if len(layer_idxs) == 0:
                    raise ValueError("decoder_side_ctc.layers must contain valid 1-indexed layer IDs")

                tgt_pad_mask = None
                if self.pad_id is not None:
                    tgt_pad_mask = y_inp.eq(int(self.pad_id))  # (B, N)

                W = self.decoder.proj.weight  # (V_ar, D)
                b = self.decoder.proj.bias
                W_ctc = W[: self.vocab_ctc]
                b_ctc = b[: self.vocab_ctc] if b is not None else None

                dec_ctc_logits_layers: list[torch.Tensor] = []
                for li in layer_idxs:
                    s = layer_states[li]  # (B, N, D)

                    if self.decoder_side_ctc_mode == "attention":
                        # Attention mode: encoder frames query decoder states
                        if self.dec_ctc_attn is None:
                            raise RuntimeError("decoder_side_ctc mode='attention' but dec_ctc_attn not initialized")
                        ctx, _ = self.dec_ctc_attn(
                            query=mem,
                            key=s,
                            value=s,
                            key_padding_mask=tgt_pad_mask,
                            need_weights=False,
                        )  # (B, Tm, D)
                        if self.dec_ctc_attn_ln is not None:
                            ctx = self.dec_ctc_attn_ln(ctx)
                    else:
                        # Pool mode: global average of decoder states, broadcast to encoder frames
                        # Mask out padding tokens if applicable
                        if tgt_pad_mask is not None:
                            mask = (~tgt_pad_mask).float().unsqueeze(-1)  # (B, N, 1)
                            s_masked = s * mask
                            dec_ctx = s_masked.sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp(min=1)
                        else:
                            dec_ctx = s.mean(dim=1, keepdim=True)  # (B, 1, D)
                        ctx = mem + dec_ctx  # (B, Tm, D) broadcast addition

                    dec_ctc_logits_layers.append(F.linear(ctx, W_ctc, b_ctc))  # (B, Tm, Vctc)

                result["dec_ctc_logits_layers"] = dec_ctc_logits_layers
                # Also return enc_lengths for CTC loss
                Tm = mem.size(1)
                if in_lengths is not None:
                    enc_lengths = torch.div(in_lengths.to(mem.device), self.encoder.ratio_ds, rounding_mode='floor')
                    enc_lengths = enc_lengths.clamp(min=1, max=Tm).long()
                else:
                    enc_lengths = torch.full((mem.size(0),), Tm, dtype=torch.long, device=mem.device)
                result["enc_lengths"] = enc_lengths

            return result

        # CTC path (per-timestep decoder)
        feats = self.encoder(x)  # (B, T', C')
        return self.decoder(feats)

    def infer(self) -> None:
        if hasattr(self.encoder, 'fuse'):
            self.encoder.fuse()
        if hasattr(self.decoder, 'recurrent'):
            self.decoder.recurrent = True

    @property
    def ratio_ds(self) -> int:
        return self.encoder.ratio_ds

    @property
    def use_ar(self) -> bool:
        return isinstance(self.decoder, ARDecoder)

    @torch.no_grad()
    def generate(self, x: torch.Tensor, len_max: int = 32) -> torch.Tensor:
        """Greedy autoregressive decoding for inference / FLOPs profiling.

        Returns the predicted token-id tensor of shape (B, len_max + 1), where
        the first column is BOS. Generation runs for exactly `len_max` steps
        without early stopping on EOS, so the cumulative cost reflects the
        worst-case trajectory.

        Encoder-output caching: the encoder (and the optional mem_proj) runs
        exactly ONCE; the loop only invokes the decoder against the cached
        memory. This is the proper inference pattern — the previous naive
        implementation re-encoded `x` every step, inflating the FLOP count
        by ~len_max times.

        Special-token IDs follow the char-tokenizer convention: PAD = self.pad_id
        (or num_cls-3 if pad_id is None), BOS = pad_id + 1.
        """
        if not self.use_ar:
            raise RuntimeError("generate() requires an AR decoder")
        B = x.size(0)
        device = x.device
        pad_id = int(self.pad_id) if self.pad_id is not None else (self.num_cls - 3)
        bos_id = pad_id + 1
        in_lengths = torch.full((B,), x.size(-1), dtype=torch.long, device=device)

        # Encode once, project to decoder dim once.
        feats, enc_pad = self._encode_with_mask(x, in_lengths)
        mem = feats if self.mem_proj is None else self.mem_proj(feats)

        y_gen = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        for _ in range(int(len_max)):
            logits = self.decoder(y_gen, mem, enc_pad)  # (B, N, V)
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            y_gen = torch.cat([y_gen, nxt], dim=1)
        return y_gen
