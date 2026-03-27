"""In-batch contrastive loss for modality alignment.

Inspired by the BC (Bottleneck Conditioning) loss from ECHWR
(Li et al., 2026), adapted for the VLM pipeline:

    c_sig  = mean-pooled connector output  (B, d_lm)
    z_text = mean-pooled GT text embedding (B, d_lm)

    s_{i,j} = tau * cos_sim(c_sig_i, z_text_j)

    L_BC = -1/(2N) sum_i [ log(exp(s_ii)/sum_j exp(s_ij))
                         + log(exp(s_ii)/sum_j exp(s_ji)) ]

Duplicate labels are filtered to avoid false negatives.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InBatchContrastiveLoss(nn.Module):
    """Symmetric InfoNCE loss between sensor and text embeddings."""

    def __init__(self, init_temperature: float = 0.07):
        super().__init__()
        # Learnable log-scale: τ = 1/temperature (CLIP convention)
        # init_temperature=0.07 → τ=14.29 → log_tau=2.66
        self.log_tau = nn.Parameter(torch.tensor(1.0 / init_temperature).log())

    @property
    def tau(self) -> float:
        return self.log_tau.exp().item()

    def forward(
        self,
        imu_tokens: torch.Tensor,
        text_emb: torch.Tensor,
        labels_mask: torch.Tensor | None = None,
        texts: list[str] | None = None,
    ) -> torch.Tensor:
        """Compute symmetric InfoNCE loss.

        Args:
            imu_tokens: (B, K, d) connector output — will be mean-pooled.
            text_emb:   (B, L, d) text embeddings — will be mean-pooled
                        (only over non-padding positions if labels_mask given).
            labels_mask: (B, L) bool mask, True = real token, False = padding.
            texts:      List of GT strings — used to filter duplicate labels.

        Returns:
            Scalar loss.
        """
        # Mean-pool to single vectors
        c_sig = imu_tokens.mean(dim=1)  # (B, d)

        if labels_mask is not None:
            # Masked mean-pool: only real tokens
            mask_f = labels_mask.unsqueeze(-1).float()  # (B, L, 1)
            z_text = (text_emb * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            z_text = text_emb.mean(dim=1)  # (B, d)

        # L2 normalize
        c_sig = F.normalize(c_sig, dim=-1)
        z_text = F.normalize(z_text, dim=-1)

        # Similarity matrix: (B, B)
        tau = self.log_tau.exp()
        logits = tau * (c_sig @ z_text.T)  # (B, B)

        B = logits.size(0)

        # Filter duplicate labels to avoid false negatives
        if texts is not None:
            # Build mask: mask[i,j] = True means i,j have same label (false negative)
            dup_mask = torch.zeros(B, B, dtype=torch.bool, device=logits.device)
            for i in range(B):
                for j in range(i + 1, B):
                    if texts[i] == texts[j]:
                        dup_mask[i, j] = True
                        dup_mask[j, i] = True
            # Set duplicate similarities to -inf so they're ignored in softmax
            if dup_mask.any():
                logits = logits.masked_fill(dup_mask, float("-inf"))

        # Symmetric InfoNCE
        target = torch.arange(B, device=logits.device)
        loss_s2t = F.cross_entropy(logits, target)       # sensor → text
        loss_t2s = F.cross_entropy(logits.T, target)     # text → sensor

        return (loss_s2t + loss_t2s) / 2
