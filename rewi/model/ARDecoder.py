# rewi/model/ARDecoder.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
    
# ---------------- Gated Attention Components ---------------- #

class ElementwiseHeadGating(nn.Module):
    """
    Head-specific *elementwise* gating after SDPA output (G1, "SDPA Elementwise" in the paper).

    y_h: (B, T, d_head) -> gate_h: sigmoid(W_h y_h + b_h) -> y_h * gate_h
    """
    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        # one small linear per head: d_head -> d_head
        self.head_linears = nn.ModuleList(
            [nn.Linear(head_dim, head_dim) for _ in range(num_heads)]
        )

        # init so gates start near 1 (don’t kill attention early)
        for lin in self.head_linears:
            nn.init.zeros_(lin.weight)
            nn.init.constant_(lin.bias, 2.0)  # sigmoid(2) ≈ 0.88

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: (B, H, T, D)
        returns: (B, H, T, D)
        """
        B, H, T, D = y.shape
        outs = []
        for h in range(H):
            yh = y[:, h]                              # (B, T, D)
            gate = torch.sigmoid(self.head_linears[h](yh))  # (B, T, D)
            outs.append(yh * gate)
        return torch.stack(outs, dim=1)               # (B, H, T, D)


class HeadwiseGating(nn.Module):
    """
    Head-specific *headwise* gating after SDPA output (G1, "SDPA Headwise" in the paper).

    y_h: (B, T, d_head) -> gate_h: sigmoid(w_h^T y_h + b_h) -> scalar per token/head.
    """
    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        # one small linear per head: d_head -> 1
        self.head_linears = nn.ModuleList(
            [nn.Linear(head_dim, 1) for _ in range(num_heads)]
        )

        for lin in self.head_linears:
            nn.init.zeros_(lin.weight)
            nn.init.constant_(lin.bias, 2.0)  # sigmoid(2) ≈ 0.88

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: (B, H, T, D)
        returns: (B, H, T, D)  with a scalar gate per (B, H, T) broadcast over D.
        """
        B, H, T, D = y.shape
        outs = []
        for h in range(H):
            yh = y[:, h]                               # (B, T, D)
            gate = torch.sigmoid(self.head_linears[h](yh))  # (B, T, 1)
            outs.append(yh * gate)                     # broadcast on D
        return torch.stack(outs, dim=1)                # (B, H, T, D)



class GatedMultiheadAttention(nn.Module):
    """
    Manual multi-head attention with SDPA output gating (G1).

    gating_type:
      - "elementwise": per-head, per-dim gating (SDPA Elementwise)
      - "headwise":   per-head scalar gating (SDPA Headwise)
    """
    def __init__(self, d_model: int, num_heads: int,
                 dropout: float = 0.0,
                 gating_type: str = "elementwise"):
        super().__init__()
        assert d_model % num_heads == 0
        assert gating_type in ("elementwise", "headwise")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.batch_first = True
        self.gating_type = gating_type

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)

        if gating_type == "elementwise":
            self.gating = ElementwiseHeadGating(num_heads, self.head_dim)
        else:  # "headwise"
            self.gating = HeadwiseGating(num_heads, self.head_dim)


    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, C) -> (B, H, T, D)
        B, T, C = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, T, D) -> (B, T, C)
        B, H, T, D = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(B, T, H * D)

    def forward(
        self,
        query: torch.Tensor,          # (B, T_q, C)
        key: torch.Tensor,            # (B, T_k, C)
        value: torch.Tensor,          # (B, T_k, C)
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T_q, _ = query.shape
        _, T_k, _ = key.shape

        q = self._split_heads(self.q_proj(query))  # (B, H, T_q, D)
        k = self._split_heads(self.k_proj(key))    # (B, H, T_k, D)
        v = self._split_heads(self.v_proj(value))  # (B, H, T_k, D)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T_q, T_k)

        # attn_mask: e.g. causal mask (T_q, T_k) bool; True = masked
        if attn_mask is not None:
            # Make sure mask is broadcastable to (B, H, T_q, T_k)
            if attn_mask.dim() == 2:        # (T_q, T_k)
                attn_mask_exp = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T_q, T_k)
            elif attn_mask.dim() == 3:      # (B, T_q, T_k)
                attn_mask_exp = attn_mask.unsqueeze(1)               # (B, 1, T_q, T_k)
            else:                            # already 4D?
                attn_mask_exp = attn_mask

            if attn_mask_exp.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask_exp, float("-inf"))
            else:
                # assume additive mask with 0 or -inf
                scores = scores + attn_mask_exp

        # key_padding_mask: (B, T_k) bool; True = masked
        if key_padding_mask is not None:
            kpm = key_padding_mask.view(B, 1, 1, T_k)  # (B, 1, 1, T_k)
            scores = scores.masked_fill(kpm, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        y = torch.matmul(attn, v)  # (B, H, T_q, D) — SDPA output per head

        # 🔥 Per-head, elementwise gating on SDPA output
        y = self.gating(y)         # (B, H, T_q, D)

        y = self._merge_heads(y)   # (B, T_q, C)
        y = self.out_proj(y)       # (B, T_q, C)
        return y


class GatedTransformerDecoderLayer(nn.Module):
    """
    Equivalent to nn.TransformerDecoderLayer with:
    - norm_first=True
    - batch_first=True
    but using GatedMultiheadAttention (G1) for both self- and cross-attention.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        norm_first: bool = True,
        gating_type: str = "elementwise",   # <--- NEW
    ):
        super().__init__()
        self.self_attn = GatedMultiheadAttention(
            d_model, nhead, dropout=dropout, gating_type=gating_type
        )
        self.multihead_attn = GatedMultiheadAttention(
            d_model, nhead, dropout=dropout, gating_type=gating_type
        )

        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # LayerNorms and residual dropouts
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.gelu

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # Self-attn: Q=K=V=x
        x = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        return self.dropout1(x)

    def _mha_block(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # Cross-attn: Q=x, K=V=mem
        x = self.multihead_attn(
            x, mem, mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        return self.dropout2(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

    def forward(
        self,
        tgt: torch.Tensor,                         # (B, T_tgt, D)
        memory: torch.Tensor,                      # (B, T_mem, D)
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        tgt_is_causal: bool = False,               # <- NEW, ignored
        memory_is_causal: bool = False,            # <- NEW, ignored
    ) -> torch.Tensor:
        x = tgt

        if self.norm_first:
            # Pre-norm: LN -> block -> residual
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            # Post-norm (not used in your setup, but included for completeness)
            x2 = self._sa_block(x, tgt_mask, tgt_key_padding_mask)
            x = self.norm1(x + x2)
            x2 = self._mha_block(x, memory, memory_mask, memory_key_padding_mask)
            x = self.norm2(x + x2)
            x2 = self._ff_block(x)
            x = self.norm3(x + x2)

        return x


# ---------------- ARDecoder with flag ---------------- #

class ARDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model: int = 256,
        nhead: int = 4,
        layers: int = 4,
        dim_ff: int = 1024,
        pdrop: float = 0.1,
        use_gated_attention: bool = False,  # <--- NEW FLAG,
        gating_type: str = "elementwise",   # <--- NEW
    ):
        super().__init__()
        self.d_model = d_model     
        self.num_layers = layers   # to use in initialization
        self.emb = nn.Embedding(vocab_size, d_model)
        self.use_gated_attention = use_gated_attention

        if use_gated_attention:
            # decoder with gated attention (elementwise/head-wise sigmoid gating on SDPA output per specific head) (paper variant)
            print(f"[ARDecoder] using gated_attention with gating_type={gating_type}")
            layer = GatedTransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_ff,
                dropout=pdrop,
                norm_first=True,
                gating_type=gating_type,  # <---
            )
        else:
            # Original vanilla PyTorch decoder layer
            print(f"[ARDecoder] using vanilla attention (no gating)")
            layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_ff,
                dropout=pdrop,
                batch_first=True,
                norm_first=True,
            )

        self.dec = nn.TransformerDecoder(layer, num_layers=layers)
        self.proj = nn.Linear(d_model, vocab_size)

        # calling custom initialization here
        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        """
        CLIP-style depth-aware initialization for the AR decoder.
        - emb: small std
        - attn Q/K/V: ~ d^{-1/2}
        - attn / MLP output projections: ~ d^{-1/2} (2L)^{-1/2}
        """
        d = self.d_model
        L = self.num_layers

        # embeddings
        nn.init.normal_(self.emb.weight, std=0.02)

        # stds
        proj_std = (d ** -0.5) * ((2 * L) ** -0.5)
        attn_std = d ** -0.5
        fc_std   = (2 * d) ** -0.5

        # iterate over decoder layers
        for layer in self.dec.layers:
            # ---- vanilla PyTorch TransformerDecoderLayer ----
            if isinstance(layer, nn.TransformerDecoderLayer):
                # self-attn
                mha = layer.self_attn
                if mha.in_proj_weight is not None:
                    nn.init.normal_(mha.in_proj_weight, std=attn_std)
                    if mha.in_proj_bias is not None:
                        nn.init.zeros_(mha.in_proj_bias)
                nn.init.normal_(mha.out_proj.weight, std=proj_std)
                if mha.out_proj.bias is not None:
                    nn.init.zeros_(mha.out_proj.bias)

                # cross-attn (if present)
                if hasattr(layer, "multihead_attn"):
                    ca = layer.multihead_attn
                    if ca.in_proj_weight is not None:
                        nn.init.normal_(ca.in_proj_weight, std=attn_std)
                        if ca.in_proj_bias is not None:
                            nn.init.zeros_(ca.in_proj_bias)
                    nn.init.normal_(ca.out_proj.weight, std=proj_std)
                    if ca.out_proj.bias is not None:
                        nn.init.zeros_(ca.out_proj.bias)

                # MLP
                nn.init.normal_(layer.linear1.weight, std=fc_std)
                if layer.linear1.bias is not None:
                    nn.init.zeros_(layer.linear1.bias)
                nn.init.normal_(layer.linear2.weight, std=proj_std)
                if layer.linear2.bias is not None:
                    nn.init.zeros_(layer.linear2.bias)

            # ---- your gated GatedTransformerDecoderLayer ----
            elif isinstance(layer, GatedTransformerDecoderLayer):
                # self- & cross-attn use GatedMultiheadAttention
                for attn in [layer.self_attn, layer.multihead_attn]:
                    nn.init.normal_(attn.q_proj.weight, std=attn_std)
                    nn.init.normal_(attn.k_proj.weight, std=attn_std)
                    nn.init.normal_(attn.v_proj.weight, std=attn_std)
                    if attn.q_proj.bias is not None:
                        nn.init.zeros_(attn.q_proj.bias)
                    if attn.k_proj.bias is not None:
                        nn.init.zeros_(attn.k_proj.bias)
                    if attn.v_proj.bias is not None:
                        nn.init.zeros_(attn.v_proj.bias)

                    nn.init.normal_(attn.out_proj.weight, std=proj_std)
                    if attn.out_proj.bias is not None:
                        nn.init.zeros_(attn.out_proj.bias)

                # MLP
                nn.init.normal_(layer.linear1.weight, std=fc_std)
                if layer.linear1.bias is not None:
                    nn.init.zeros_(layer.linear1.bias)
                nn.init.normal_(layer.linear2.weight, std=proj_std)
                if layer.linear2.bias is not None:
                    nn.init.zeros_(layer.linear2.bias)

        # final projection (CLS-equivalent for decoder outputs)
        nn.init.normal_(self.proj.weight, std=d ** -0.5)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, y_inp, memory, mem_pad_mask=None):
        # y_inp: (B, N) with <bos> at 0; memory: (B, Tm, D)
        tgt = self.emb(y_inp)  # (B, N, D)
        N = y_inp.size(1)

        # Causal mask: True = masked (upper triangle)
        causal = torch.triu(
            torch.ones(N, N, device=y_inp.device, dtype=torch.bool),
            diagonal=1,
        )

        h = self.dec(
            tgt,
            memory,
            tgt_mask=causal,
            memory_key_padding_mask=mem_pad_mask,
        )
        return self.proj(h)  # (B, N, V)
