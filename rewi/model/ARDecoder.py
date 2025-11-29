# rewi/model/ARDecoder.py
import torch
import torch.nn as nn

class ARDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, layers=4, dim_ff=1024, pdrop=0.1):
        super().__init__()
        self.d_model = d_model  # <-- add this
        self.emb = nn.Embedding(vocab_size, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_ff, dropout=pdrop, batch_first=True, norm_first=True
        )
        self.dec = nn.TransformerDecoder(layer, num_layers=layers)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, y_inp, memory, mem_pad_mask=None):
        # y_inp: (B, N) with <bos> at 0; memory: (B, Tm, D)
        tgt = self.emb(y_inp)                                  # (B, N, D)
        N = y_inp.size(1)
        causal = torch.triu(torch.ones(N, N, device=y_inp.device, dtype=torch.bool), 1)
        h = self.dec(tgt, memory, tgt_mask=causal, memory_key_padding_mask=mem_pad_mask)
        return self.proj(h)                                    # (B, N, V)
