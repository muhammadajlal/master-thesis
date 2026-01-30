import torch
import torch.nn as nn

from .ARDecoder import ARDecoder
from .builders import build_decoder, build_encoder


class BaseModel(nn.Module):
    """
    Supports both:
      - CTC pipeline (encoder + per-timestep decoder)
      - AR pipeline (encoder + ARDecoder with cross-attention)
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
    ) -> None:
        super().__init__()
        self.arch_en = arch_en
        self.arch_de = arch_de
        self.in_chan = in_chan
        self.num_cls = num_cls
        self.len_seq = len_seq

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
    ):
        # AR path (teacher forcing or dummy path for profiling)
        if isinstance(self.decoder, ARDecoder):
            mem, enc_pad = self._encode_with_mask(x, in_lengths)
            if self.mem_proj is not None:
                mem = self.mem_proj(mem)

            if y_inp is None:
                B = mem.size(0)
                y_inp = torch.zeros(B, 1, dtype=torch.long, device=mem.device)

            return self.decoder(y_inp, mem, enc_pad)  # (B, N, V)

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
