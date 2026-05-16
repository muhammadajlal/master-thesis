import torch
import torch.nn as nn
import torch.nn.functional as F

from .ARDecoder import ARDecoder
from .builders import build_decoder, build_encoder


class DualHeadModel(nn.Module):
    """Shared encoder with two heads:

    - AR head: cross-attention decoder trained with CE
    - CTC head: per-timestep classifier trained with CTC

    Notes:
    - The AR vocab and CTC vocab are allowed to differ.
    - CTC head outputs logits; rewi.loss.CTCLoss applies log_softmax.
    """

    def __init__(
        self,
        arch_en: str,
        arch_ar: str,
        arch_ctc: str,
        in_chan: int,
        vocab_ar: int,
        vocab_ctc: int,
        len_seq: int = 0,
        *,
        use_gated_attention: bool = False,
        gating_type: str = "elementwise",
        pad_id: int | None = None,
        bos_id: int | None = None,
        eos_id: int | None = None,
        tie_cfg: dict | None = None,
        dual_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        self.arch_en = arch_en
        self.arch_ar = arch_ar
        self.arch_ctc = arch_ctc
        self.in_chan = in_chan
        self.vocab_ar = vocab_ar
        self.vocab_ctc = vocab_ctc
        self.len_seq = len_seq

        tie_cfg = tie_cfg or {}
        self.tie_ctc_to_ar_outproj = bool(tie_cfg.get("ctc_to_ar_outproj", False))
        self.tie_ar_emb_outproj = bool(tie_cfg.get("ar_emb_outproj", False))
        self.ctc_input_space = str(tie_cfg.get("ctc_input_space", "enc")).lower()

        # Optional IDs (used for sanity checks/logging only)
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

        dual_cfg = dual_cfg or {}

        # Optional auxiliary supervision inside the AR decoder.
        # - deep_supervision: token CE at intermediate decoder layers (no extra classifier params)
        # - decoder_side_ctc: CTC on encoder time axis conditioned on intermediate decoder states
        self.deep_supervision_cfg = dual_cfg.get("deep_supervision", {}) or {}
        self.decoder_side_ctc_cfg = dual_cfg.get("decoder_side_ctc", {}) or {}
        self.deep_supervision_enabled = bool(self.deep_supervision_cfg.get("enabled", False))
        self.decoder_side_ctc_enabled = bool(self.decoder_side_ctc_cfg.get("enabled", False))
        self.decoder_side_ctc_mode = str(self.decoder_side_ctc_cfg.get("mode", "attention")).lower()  # "attention" or "pool"

        self.encoder = build_encoder(in_chan, arch_en, len_seq)

        # AR decoder
        self.decoder = build_decoder(
            self.encoder.dim_out,
            vocab_ar,
            arch_ar,
            len_seq // self.encoder.ratio_ds if arch_en != 'trans' else 0,
            use_gated_attention=use_gated_attention,
            gating_type=gating_type,
        )
        if not isinstance(self.decoder, ARDecoder):
            raise ValueError(
                f"DualHeadModel expects an AR decoder arch for arch_ar, got: {arch_ar}"
            )

        # Optional secondary ablation: tie AR token embedding matrix to the AR output projection.
        # This is LM-style tying and is independent of the CTC↔AR tying.
        if self.tie_ar_emb_outproj:
            if not (hasattr(self.decoder, "emb") and hasattr(self.decoder, "proj")):
                raise ValueError("AR embedding/out-projection tying requested but decoder lacks emb/proj.")
            if getattr(self.decoder.emb, "weight", None) is None or getattr(self.decoder.proj, "weight", None) is None:
                raise ValueError("AR embedding/out-projection tying requested but emb/proj weights missing.")
            if self.decoder.proj.weight.shape != self.decoder.emb.weight.shape:
                raise ValueError(
                    "AR embedding/out-projection tying requires proj.weight and emb.weight to have the same shape, "
                    f"got proj={tuple(self.decoder.proj.weight.shape)}, emb={tuple(self.decoder.emb.weight.shape)}"
                )
            # Share parameters (no new weights). Bias (if present) stays separate.
            self.decoder.proj.weight = self.decoder.emb.weight

        self.mem_proj = None
        dec_dim = self.decoder.d_model
        enc_dim = self.encoder.dim_out
        if enc_dim != dec_dim:
            self.mem_proj = nn.Linear(enc_dim, dec_dim)

        # Decoder-side CTC: "attention" mode uses reverse cross-attention (encoder frames query decoder states)
        # "pool" mode uses global average of decoder states (zero extra params)
        self.dec_ctc_attn = None
        self.dec_ctc_attn_ln = None
        if self.decoder_side_ctc_enabled:
            if self.decoder_side_ctc_mode == "attention":
                nhead = int(self.decoder_side_ctc_cfg.get("nhead", getattr(self.decoder, "nhead", 4)))
                if dec_dim % nhead != 0:
                    raise ValueError(
                        f"decoder_side_ctc.nhead must divide decoder dim. Got d_dec={dec_dim}, nhead={nhead}" 
                    )
                self.dec_ctc_attn = nn.MultiheadAttention(embed_dim=dec_dim, num_heads=nhead, batch_first=True)
                if bool(self.decoder_side_ctc_cfg.get("layernorm", True)):
                    self.dec_ctc_attn_ln = nn.LayerNorm(dec_dim)
            elif self.decoder_side_ctc_mode != "pool":
                raise ValueError(f"decoder_side_ctc.mode must be 'attention' or 'pool', got: {self.decoder_side_ctc_mode}")

        # CTC head
        if self.tie_ctc_to_ar_outproj:
            # Variant A: no separate CTC classifier weights; reuse AR decoder classifier rows.
            if arch_ctc not in {"linear", "lin"}:
                raise ValueError(
                    "dual_head.tie.ctc_to_ar_outproj=true only supports arch_ctc='linear'"
                )
            if self.ctc_input_space not in {"enc", "dec"}:
                raise ValueError(
                    f"dual_head.tie.ctc_input_space must be 'enc' or 'dec', got: {self.ctc_input_space}"
                )
            if vocab_ar < vocab_ctc:
                raise ValueError(
                    f"CTC↔AR tying requires vocab_ar >= vocab_ctc, got vocab_ar={vocab_ar}, vocab_ctc={vocab_ctc}"
                )
            if self.ctc_input_space == "enc" and enc_dim != dec_dim:
                raise ValueError(
                    "CTC↔AR tying with ctc_input_space='enc' requires enc_dim == dec_dim. "
                    "Use ctc_input_space='dec' to reuse mem_proj for dimension matching."
                )

            # Hard layout asserts (character-mode expected): [blank+chars] + [PAD,BOS,EOS] appended for AR.
            # These IDs are provided by setup_tokenizer (main.py).
            if pad_id is not None and bos_id is not None and eos_id is not None:
                expected_pad = int(vocab_ctc)
                expected_bos = int(vocab_ctc + 1)
                expected_eos = int(vocab_ctc + 2)
                if int(vocab_ar) == int(vocab_ctc + 3):
                    if not (int(pad_id) == expected_pad and int(bos_id) == expected_bos and int(eos_id) == expected_eos):
                        raise ValueError(
                            "CTC↔AR tying expects AR IDs layout PAD=vocab_ctc, BOS=vocab_ctc+1, EOS=vocab_ctc+2 "
                            f"when vocab_ar=vocab_ctc+3. Got PAD={pad_id}, BOS={bos_id}, EOS={eos_id}, "
                            f"vocab_ctc={vocab_ctc}, vocab_ar={vocab_ar}."
                        )

            self.ctc_head = None
        else:
            if arch_ctc in {"linear", "lin"}:
                self.ctc_head = nn.Linear(self.encoder.dim_out, vocab_ctc)
            else:
                self.ctc_head = build_decoder(
                    self.encoder.dim_out,
                    vocab_ctc,
                    arch_ctc,
                    len_seq // self.encoder.ratio_ds if arch_en != 'trans' else 0,
                )

    def compute_ctc_logits(self, feats: torch.Tensor) -> torch.Tensor:
        """Compute per-timestep CTC logits.

        feats: (B, Tm, Cenc)
        returns: (B, Tm, Vctc)
        """
        if not self.tie_ctc_to_ar_outproj:
            if self.ctc_head is None:
                raise RuntimeError("CTC head is not initialized.")
            return self.ctc_head(feats)

        # Variant A: share AR decoder classifier (self.decoder.proj) for the shared prefix.
        h = feats
        if self.ctc_input_space == "dec" and self.mem_proj is not None:
            h = self.mem_proj(h)

        W = self.decoder.proj.weight  # (V_ar, D)
        b = self.decoder.proj.bias
        W_ctc = W[: self.vocab_ctc]
        b_ctc = b[: self.vocab_ctc] if b is not None else None
        return F.linear(h, W_ctc, b_ctc)

    def _encode_with_mask(self, x: torch.Tensor, in_lengths: torch.Tensor | None):
        if in_lengths is None:
            with torch.no_grad():
                valid = (x.abs().sum(dim=1) > 1e-6)
                in_lengths = valid.sum(dim=1)
        else:
            in_lengths = in_lengths.to(device=x.device)

        feats = self.encoder(x)  # (B, Tm, Cenc)
        Tm = feats.size(1)
        enc_lengths = torch.div(in_lengths, self.encoder.ratio_ds, rounding_mode='floor')
        enc_lengths = enc_lengths.clamp(min=1, max=Tm)
        enc_lengths = enc_lengths.to(device=feats.device, dtype=torch.long)
        enc_pad = torch.arange(Tm, device=feats.device).unsqueeze(0) >= enc_lengths.unsqueeze(1)
        return feats, enc_pad, enc_lengths

    def forward(
        self,
        x: torch.Tensor,
        *,
        in_lengths: torch.Tensor | None = None,
        y_inp: torch.Tensor | None = None,
        return_ar: bool = True,
        return_ctc: bool = True,
    ) -> dict:
        feats, enc_pad, enc_lengths = self._encode_with_mask(x, in_lengths)

        out: dict[str, torch.Tensor] = {"enc_lengths": enc_lengths}

        if return_ctc:
            out["ctc_logits"] = self.compute_ctc_logits(feats)  # (B, Tm, Vctc)

        if return_ar:
            mem = feats
            if self.mem_proj is not None:
                mem = self.mem_proj(mem)

            if y_inp is None:
                B = mem.size(0)
                y_inp = torch.zeros(B, 1, dtype=torch.long, device=mem.device)

            need_layer_states = bool(self.training and (self.deep_supervision_enabled or self.decoder_side_ctc_enabled))
            if need_layer_states:
                dec_out = self.decoder(y_inp, mem, enc_pad, return_layer_states=True)
                out["ar_logits"] = dec_out["logits"]  # (B, N, Var)
                out["ar_layer_states"] = dec_out["layer_states"]  # list[(B, N, D)]

                # Optional: logits from intermediate decoder layers (for deep supervision).
                if self.deep_supervision_enabled:
                    out["ar_logits_layers"] = [self.decoder.proj(s) for s in out["ar_layer_states"]]

                # Optional: decoder-conditioned CTC logits on encoder time axis (T')
                if self.decoder_side_ctc_enabled:
                    if self.dec_ctc_attn is None:
                        raise RuntimeError("decoder_side_ctc enabled but dec_ctc_attn is not initialized")

                    # Which decoder layers to use (1-indexed in config)
                    layers_cfg = self.decoder_side_ctc_cfg.get("layers", [1, 2, 3])
                    if not isinstance(layers_cfg, (list, tuple)):
                        layers_cfg = [layers_cfg]
                    layer_idxs = [int(l) - 1 for l in layers_cfg]
                    layer_idxs = [i for i in layer_idxs if i >= 0]
                    if len(layer_idxs) == 0:
                        raise ValueError("decoder_side_ctc.layers must contain valid 1-indexed layer IDs")
                    if max(layer_idxs) >= len(out["ar_layer_states"]):
                        raise ValueError(
                            f"decoder_side_ctc.layers out of range. Got {layers_cfg}, decoder has {len(out['ar_layer_states'])} layers"
                        )

                    tgt_pad_mask = None
                    if self.pad_id is not None:
                        tgt_pad_mask = y_inp.eq(int(self.pad_id))  # (B, N)

                    W = self.decoder.proj.weight  # (V_ar, D)
                    b = self.decoder.proj.bias
                    W_ctc = W[: self.vocab_ctc]
                    b_ctc = b[: self.vocab_ctc] if b is not None else None

                    dec_ctc_logits_layers: list[torch.Tensor] = []
                    for li in layer_idxs:
                        s = out["ar_layer_states"][li]  # (B, N, D)

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
                            if tgt_pad_mask is not None:
                                mask = (~tgt_pad_mask).float().unsqueeze(-1)  # (B, N, 1)
                                s_masked = s * mask
                                dec_ctx = s_masked.sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp(min=1)
                            else:
                                dec_ctx = s.mean(dim=1, keepdim=True)  # (B, 1, D)
                            ctx = mem + dec_ctx  # (B, Tm, D) broadcast addition

                        dec_ctc_logits_layers.append(F.linear(ctx, W_ctc, b_ctc))  # (B, Tm, Vctc)

                    out["dec_ctc_logits_layers"] = dec_ctc_logits_layers
            else:
                out["ar_logits"] = self.decoder(y_inp, mem, enc_pad)  # (B, N, Var)

        return out

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
        return True  # DualHeadModel always has an AR decoder

    @torch.no_grad()
    def generate(self, x: torch.Tensor, len_max: int = 32) -> torch.Tensor:
        """Greedy AR generation through the AR head only (CTC head skipped).

        Encoder-output caching: the encoder (and optional mem_proj) runs
        exactly ONCE; the loop only invokes the AR decoder against the cached
        memory. CTC head is not computed.

        Runs for exactly ``len_max`` steps without EOS early-stop.
        """
        B = x.size(0)
        device = x.device
        pad_id = int(self.pad_id) if self.pad_id is not None else (self.vocab_ar - 3)
        bos_id = int(self.bos_id) if self.bos_id is not None else (pad_id + 1)
        in_lengths = torch.full((B,), x.size(-1), dtype=torch.long, device=device)

        # Encode once, project to decoder dim once.
        feats, enc_pad, _ = self._encode_with_mask(x, in_lengths)
        mem = feats if self.mem_proj is None else self.mem_proj(feats)

        y_gen = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        for _ in range(int(len_max)):
            logits = self.decoder(y_gen, mem, enc_pad)  # (B, N, V_ar)
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            y_gen = torch.cat([y_gen, nxt], dim=1)
        return y_gen
