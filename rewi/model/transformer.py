# decoder/transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Transformer"]

class Transformer(nn.Module):
    """Transformer encoder for per-timestep classification (CTC-ready).

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
        self.pos = nn.Parameter(torch.zeros(1, pe_max_len, d_model))  # learned PE

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
            raise ValueError(f"T={T} exceeds pe_max_len={self.pos.size(1)}; increase it or swap to sinusoidal PE.")

        # Optional masking (if you ever pass lengths later). If not provided, no mask is used.
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
