"""Auxiliary losses for bridging the modality gap in VLM training.

Implements five approaches:

K1 — CTC Compression + Per-Token MSE Distillation
    Compress encoder output via CTC best-path (remove blanks + merge repeats),
    project to d_lm, then MSE against GPT-2 text embeddings per position.

K2 — LegoSLM CTC Posterior Embedding Reconstruction
    Reconstruct connector output as weighted sum of GPT-2 vocabulary embeddings
    using CTC posteriors. Modality gap is impossible by construction.
    (Implemented as a new connector in projectors.py)

K3 — ECHWR Error-Based Contrastive (EC) Loss
    Generate hard negatives via single-character edits (sub/ins/del),
    encode with a small text encoder, contrastive loss forces fine-grained
    discrimination.

K4 — SEA Token-Level Contrastive Loss
    Per-position contrastive loss using CTC alignment to assign character
    labels to each IMU token. Preserves sequential structure.

K5 — Soft KV Injection
    (Implemented as model modification in vlm_model.py)
"""
from __future__ import annotations

import random
import string
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
# K1: CTC Compression + Per-Token MSE Distillation
# ═══════════════════════════════════════════════════════════════════

def ctc_compress(
    enc_states: torch.Tensor,
    ctc_logits: torch.Tensor,
    target_len: int,
    blank_id: int = 0,
) -> torch.Tensor:
    """Compress encoder states via CTC best-path decoding.

    1. Take argmax of ctc_logits → best-path sequence
    2. Remove blank tokens and merge consecutive repeats
    3. Gather the corresponding encoder frames
    4. Pad or truncate to target_len

    Args:
        enc_states: (T, d_enc) encoder output for one sample.
        ctc_logits: (T, V_ctc) CTC logits for one sample.
        target_len: desired output length (= number of target characters).
        blank_id: CTC blank index (default 0).

    Returns:
        (target_len, d_enc) compressed encoder states.
    """
    best_path = ctc_logits.argmax(dim=-1)  # (T,)
    device = enc_states.device

    # Collect indices of non-blank, non-repeat frames
    keep_indices = []
    prev = -1
    for t in range(best_path.size(0)):
        token = best_path[t].item()
        if token != blank_id and token != prev:
            keep_indices.append(t)
        prev = token

    if len(keep_indices) == 0:
        # Fallback: uniform sample
        keep_indices = list(range(0, enc_states.size(0), max(1, enc_states.size(0) // max(target_len, 1))))

    # Gather frames
    idx = torch.tensor(keep_indices, device=device, dtype=torch.long)
    compressed = enc_states[idx]  # (N, d_enc)

    # Pad or truncate to target_len
    N = compressed.size(0)
    if N >= target_len:
        compressed = compressed[:target_len]
    else:
        pad = torch.zeros(target_len - N, compressed.size(1), device=device)
        compressed = torch.cat([compressed, pad], dim=0)

    return compressed  # (target_len, d_enc)


class CTCCompressMSELoss(nn.Module):
    """K1: Per-token MSE between CTC-compressed encoder features and GPT-2 text embeddings.

    Requires a projection layer from d_enc to d_lm.
    """

    def __init__(self, d_enc: int, d_lm: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_enc, d_lm),
            nn.LayerNorm(d_lm),
        )

    def forward(
        self,
        enc_states: torch.Tensor,   # (B, T, d_enc)
        ctc_logits: torch.Tensor,    # (B, T, V_ctc)
        text_emb: torch.Tensor,      # (B, L, d_lm)
        labels_mask: torch.Tensor,   # (B, L) True = real token
    ) -> torch.Tensor:
        """Compute per-token MSE loss.

        For each sample, compress encoder via CTC best-path to match
        the number of real text tokens, project to d_lm, then MSE.
        """
        B = enc_states.size(0)
        total_loss = 0.0
        count = 0

        for i in range(B):
            mask_i = labels_mask[i]  # (L,)
            n_tokens = mask_i.sum().item()
            if n_tokens == 0:
                continue

            # Compress encoder for this sample
            compressed = ctc_compress(
                enc_states[i], ctc_logits[i],
                target_len=int(n_tokens),
            )  # (n_tokens, d_enc)

            # Project to d_lm
            projected = self.proj(compressed)  # (n_tokens, d_lm)

            # Get real text embeddings
            real_text_emb = text_emb[i][mask_i]  # (n_tokens, d_lm)

            # MSE per token
            total_loss = total_loss + F.mse_loss(projected, real_text_emb.detach())
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=enc_states.device, requires_grad=True)
        return total_loss / count


# ═══════════════════════════════════════════════════════════════════
# K3: ECHWR Error-Based Contrastive (EC) Loss
# ═══════════════════════════════════════════════════════════════════

class MiniTextEncoder(nn.Module):
    """Small Transformer encoder for text sequences (for EC loss).

    Lightweight Transformer encoder, d_model=128, projects to d_out.
    """

    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, d_out: int = 768, max_len: int = 128):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, d_out)
        self.ln = nn.LayerNorm(d_out)

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode text tokens to a single vector.

        Args:
            token_ids: (B, L) integer token IDs.
            mask: (B, L) True = real token.

        Returns:
            (B, d_out) L2-normalized text representation.
        """
        x = self.embedding(token_ids)  # (B, L, d_model)
        L = x.size(1)
        x = x + self.pos_embed[:, :L, :]
        x = self.encoder(x)  # (B, L, d_model)

        # Mean-pool over real tokens
        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()  # (B, L, 1)
            x = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        x = self.ln(self.proj(x))  # (B, d_out)
        return F.normalize(x, dim=-1)


def generate_hard_negatives(text: str, n_neg: int = 4) -> list[str]:
    """Generate hard negatives via single-character edits.

    Operations: substitution, deletion, insertion.
    Each negative differs by exactly one character from the original.
    """
    alphabet = string.ascii_letters + "äöüÄÖÜß"
    negatives = []
    chars = list(text)
    if len(chars) == 0:
        return [random.choice(alphabet) for _ in range(n_neg)]

    for _ in range(n_neg * 3):  # generate extra, deduplicate
        if len(negatives) >= n_neg:
            break
        op = random.randint(0, 2)
        pos = random.randint(0, max(0, len(chars) - 1))

        if op == 0:  # substitution
            new_chars = list(chars)
            new_char = random.choice(alphabet)
            while new_char == chars[pos] and len(alphabet) > 1:
                new_char = random.choice(alphabet)
            new_chars[pos] = new_char
            neg = "".join(new_chars)
        elif op == 1:  # deletion
            if len(chars) > 1:
                new_chars = list(chars)
                del new_chars[pos]
                neg = "".join(new_chars)
            else:
                continue
        else:  # insertion
            new_chars = list(chars)
            new_chars.insert(pos, random.choice(alphabet))
            neg = "".join(new_chars)

        if neg != text and neg not in negatives:
            negatives.append(neg)

    # Fill if not enough
    while len(negatives) < n_neg:
        neg_chars = list(chars)
        pos = random.randint(0, len(neg_chars) - 1)
        neg_chars[pos] = random.choice(alphabet)
        negatives.append("".join(neg_chars))

    return negatives[:n_neg]


class ECContrastiveLoss(nn.Module):
    """K3: ECHWR-style Error-based Contrastive loss.

    Uses a MiniTextEncoder to encode GT and hard-negative texts.
    IMU representation must be more similar to GT than to negatives.
    """

    def __init__(self, vocab_size: int, d_lm: int = 768, n_negatives: int = 4,
                 temperature: float = 0.07):
        super().__init__()
        self.text_encoder = MiniTextEncoder(
            vocab_size=vocab_size, d_model=128, d_out=d_lm,
        )
        self.n_negatives = n_negatives
        self.log_tau = nn.Parameter(torch.tensor(1.0 / temperature).log())
        self.d_lm = d_lm

    def forward(
        self,
        imu_tokens: torch.Tensor,    # (B, K, d_lm) connector output
        texts: list[str],              # GT strings
        char_to_id: dict[str, int],    # char → ID mapping for text encoder
        pad_id: int = 0,
    ) -> torch.Tensor:
        """Compute EC contrastive loss.

        For each sample: encode GT + N negatives with text encoder,
        compute similarity with mean-pooled IMU representation,
        cross-entropy targets the GT.
        """
        B = imu_tokens.size(0)
        device = imu_tokens.device
        tau = self.log_tau.exp()

        # Mean-pool IMU tokens → (B, d_lm), L2 normalize
        imu_repr = F.normalize(imu_tokens.mean(dim=1), dim=-1)  # (B, d_lm)

        total_loss = 0.0

        for i in range(B):
            gt_text = texts[i]
            negatives = generate_hard_negatives(gt_text, self.n_negatives)
            all_texts = [gt_text] + negatives  # 1 + N texts

            # Tokenize all texts
            max_len = max(len(t) for t in all_texts)
            token_ids = torch.full((len(all_texts), max_len), pad_id,
                                   dtype=torch.long, device=device)
            token_mask = torch.zeros(len(all_texts), max_len,
                                     dtype=torch.bool, device=device)
            for j, txt in enumerate(all_texts):
                ids = [char_to_id.get(c, pad_id) for c in txt]
                token_ids[j, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
                token_mask[j, :len(ids)] = True

            # Encode all texts → (1+N, d_lm)
            text_repr = self.text_encoder(token_ids, token_mask)  # (1+N, d_lm)

            # Similarities: (1+N,)
            sims = tau * (imu_repr[i:i+1] @ text_repr.T).squeeze(0)  # (1+N,)

            # Target: index 0 is GT
            target = torch.tensor([0], device=device)
            total_loss = total_loss + F.cross_entropy(sims.unsqueeze(0), target)

        return total_loss / max(B, 1)


# ═══════════════════════════════════════════════════════════════════
# K4: SEA Token-Level Contrastive Loss
# ═══════════════════════════════════════════════════════════════════

def ctc_force_align(
    ctc_logits: torch.Tensor,
    blank_id: int = 0,
) -> torch.Tensor:
    """Simple CTC alignment: assign each frame to a character via best-path.

    Returns:
        (T,) tensor of character indices (0-based, excluding blank).
        Frames assigned to blank get -1.
    """
    best_path = ctc_logits.argmax(dim=-1)  # (T,)
    T = best_path.size(0)
    alignment = torch.full((T,), -1, dtype=torch.long, device=best_path.device)

    char_idx = -1
    prev = -1
    for t in range(T):
        token = best_path[t].item()
        if token == blank_id:
            # Blank frame — assign to nearest character
            if char_idx >= 0:
                alignment[t] = char_idx
        elif token != prev:
            char_idx += 1
            alignment[t] = char_idx
        else:
            alignment[t] = char_idx
        prev = token

    return alignment


class SEATokenContrastiveLoss(nn.Module):
    """K4: Token-level contrastive loss (SEA-inspired).

    For each IMU token, use CTC alignment to determine its character label.
    Then contrastive loss: each IMU token should be closest to its
    corresponding character's GPT-2 embedding among all characters in batch.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor(1.0 / temperature).log())

    def forward(
        self,
        imu_tokens: torch.Tensor,    # (B, K, d_lm) connector output
        enc_states: torch.Tensor,     # (B, T, d_enc) raw encoder (unused here)
        ctc_logits: torch.Tensor,     # (B, T, V_ctc)
        text_emb: torch.Tensor,       # (B, L, d_lm) text embeddings
        labels_mask: torch.Tensor,    # (B, L)
    ) -> torch.Tensor:
        """Per-position contrastive loss using CTC alignment.

        Strategy: for each connector output position k, find which character
        it corresponds to via CTC alignment on the encoder, then contrastive
        loss against that character's text embedding.
        """
        B, K, d = imu_tokens.shape
        T = ctc_logits.size(1)
        device = imu_tokens.device
        tau = self.log_tau.exp()

        # Collect all unique character embeddings across batch as negatives
        all_char_embs = []
        for i in range(B):
            mask_i = labels_mask[i]
            if mask_i.any():
                all_char_embs.append(text_emb[i][mask_i])  # (n_i, d)
        if not all_char_embs:
            return torch.tensor(0.0, device=device, requires_grad=True)

        char_bank = torch.cat(all_char_embs, dim=0)  # (N_total, d)
        char_bank = F.normalize(char_bank, dim=-1)

        total_loss = 0.0
        count = 0

        for i in range(B):
            mask_i = labels_mask[i]
            n_chars = mask_i.sum().item()
            if n_chars == 0:
                continue

            real_text_emb = text_emb[i][mask_i]  # (n_chars, d)

            # CTC alignment on encoder frames
            alignment = ctc_force_align(ctc_logits[i])  # (T,)

            # Map encoder alignment to connector positions
            # If connector has K tokens from T encoder frames, scale alignment
            scale = T / K
            connector_alignment = torch.full((K,), -1, dtype=torch.long, device=device)
            for k in range(K):
                # Find the encoder frame this connector position maps to
                t_center = int(k * scale + scale / 2)
                t_center = min(t_center, T - 1)
                connector_alignment[k] = alignment[t_center]

            # For each connector position with valid alignment, contrastive loss
            imu_k = F.normalize(imu_tokens[i], dim=-1)  # (K, d)

            for k in range(K):
                char_id = connector_alignment[k].item()
                if char_id < 0 or char_id >= n_chars:
                    continue

                # Positive: the aligned character embedding
                pos_emb = F.normalize(real_text_emb[char_id:char_id+1], dim=-1)  # (1, d)

                # Similarities against full char bank
                sims = tau * (imu_k[k:k+1] @ char_bank.T)  # (1, N_total)

                # Similarity with positive
                pos_sim = tau * (imu_k[k:k+1] @ pos_emb.T)  # (1, 1)

                # InfoNCE: log(exp(pos) / (exp(pos) + sum(exp(neg))))
                # Use log_softmax over [pos, all_bank]
                all_sims = torch.cat([pos_sim, sims], dim=-1)  # (1, 1+N_total)
                target = torch.zeros(1, dtype=torch.long, device=device)
                total_loss = total_loss + F.cross_entropy(all_sims, target)
                count += 1

        if count == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return total_loss / count
