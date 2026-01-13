import torch
import torch.nn as nn
from torchaudio.models import Conformer

class ConformerEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 13,
        d_model: int = 256,
        num_layers: int = 12,
        num_heads: int = 4,
        ffn_dim: int = 1024,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        subsample: int = 4,
    ):
        super().__init__()
        if subsample not in (1, 2, 4):
            raise ValueError("subsample must be one of {1,2,4}")
        stride = subsample

        self.frontend = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        self.conformer = Conformer(
            input_dim=d_model,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=conv_kernel_size,
            dropout=dropout,
        )

        self.d_model = d_model
        self.subsample = subsample

        self.dim_out = d_model        # ✅ ADDED: what BaseModel expects
        self.ratio_ds = subsample     # ✅ ADDED: downsample factor for BaseModel

    # rewi/model/encoder/conformer_en.py

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)  -> return (B, T', D)

        with torch.no_grad():
            valid = (x.abs().sum(dim=1) > 1e-6)
            in_lengths = valid.sum(dim=1).to(torch.int64)

        x = self.frontend(x)  # (B, D, T')

        if self.subsample > 1:
            enc_lengths = torch.div(in_lengths, self.subsample, rounding_mode='floor')
        else:
            enc_lengths = in_lengths
        Tprime = x.size(-1)
        enc_lengths = enc_lengths.clamp_(min=1, max=Tprime).to(x.device)   # ✅ ADDED: keep on same device

        max_len = int(enc_lengths.max().item())
        if max_len < Tprime:
            x = x[:, :, :max_len]                                          # ✅ already added earlier

        x = x.transpose(1, 2)  # (B, T', D)

        out = self.conformer(x, lengths=enc_lengths)                       # (B, T', D)  or  ((B,T',D), lengths)

        # ✅ ADDED: handle both return signatures
        if isinstance(out, tuple):
            x = out[0]
        else:
            x = out

        return x


