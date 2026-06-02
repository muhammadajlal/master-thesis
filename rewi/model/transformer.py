# decoder/transformer.py
import math
import torch
import torch.nn as nn

__all__ = ["Transformer"]


def _sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, max_len, d_model)


class Transformer(nn.Module):
    """Transformer encoder for per-timestep classification (CTC-ready).

    Uses sinusoidal positional encoding (param-free) so the parameter count
    mirrors the AR decoder counterpart, which itself carries no learned
    positional buffer. The previous learned-PE implementation added
    pe_max_len * d_model parameters to every CTC variant, breaking the
    intended parameter match with ar_transformer_* siblings.

    Input:  x -> (B, T, C)
    Output: logits (B, T, V) by default (apply_softmax=False)
    """
    def __init__(
        self,
        size_in: int,
        num_cls: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_ff: int = 1024,
        p_drop: float = 0.1,
        pe_max_len: int = 4096,
        apply_softmax: bool = True,  # keep False for CTC loss
    ) -> None:
        super().__init__()
        self.apply_softmax = apply_softmax

        self.inp = nn.Linear(size_in, d_model)
        self.register_buffer("pos", _sinusoidal_pe(pe_max_len, d_model), persistent=False)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=p_drop, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc  = nn.Linear(d_model, num_cls)
        if self.apply_softmax:
            self.softmax = nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape
        if T > self.pos.size(1):
            raise ValueError(f"T={T} exceeds pe_max_len={self.pos.size(1)}; increase it.")

        pad_mask = None
        if lengths is not None:
            lengths = lengths.to(dtype=torch.long, device=x.device)
            pad_mask = torch.arange(T, device=x.device)[None, :] >= lengths[:, None]  # True=PAD

        h = self.inp(x) + self.pos[:, :T]                    # (B, T, D)
        h = self.enc(h, src_key_padding_mask=pad_mask)       # (B, T, D)
        logits = self.fc(h)                                  # (B, T, V)

        if self.apply_softmax:
            return self.softmax(logits)
        return logits
