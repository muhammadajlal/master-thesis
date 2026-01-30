import torch
import torch.nn as nn

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
    ) -> None:
        super().__init__()
        self.arch_en = arch_en
        self.arch_ar = arch_ar
        self.arch_ctc = arch_ctc
        self.in_chan = in_chan
        self.vocab_ar = vocab_ar
        self.vocab_ctc = vocab_ctc
        self.len_seq = len_seq

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

        self.mem_proj = None
        dec_dim = self.decoder.d_model
        enc_dim = self.encoder.dim_out
        if enc_dim != dec_dim:
            self.mem_proj = nn.Linear(enc_dim, dec_dim)

        # CTC head
        if arch_ctc in {"linear", "lin"}:
            self.ctc_head = nn.Linear(self.encoder.dim_out, vocab_ctc)
        else:
            self.ctc_head = build_decoder(
                self.encoder.dim_out,
                vocab_ctc,
                arch_ctc,
                len_seq // self.encoder.ratio_ds if arch_en != 'trans' else 0,
            )

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
            out["ctc_logits"] = self.ctc_head(feats)  # (B, Tm, Vctc)

        if return_ar:
            mem = feats
            if self.mem_proj is not None:
                mem = self.mem_proj(mem)

            if y_inp is None:
                B = mem.size(0)
                y_inp = torch.zeros(B, 1, dtype=torch.long, device=mem.device)

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
