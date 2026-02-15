"""
Encoder feature extraction for comparing CNN representations across models.

Usage:
    extract_encoder_features(model, dataloader, device, categories)
    -> returns dict with 'features', 'char_labels', 'word_labels', 'frame_indices'
"""
import os
import re

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader


def _migrate_legacy_mha_keys(state_dict: dict) -> dict:
    """Migrate old separate Q/K/V projection keys to nn.MultiheadAttention format.

    Old: *.self_attn.q_proj.weight  ->  New: *.self_attn.mha.in_proj_weight (concat Q,K,V)
    Also handles multihead_attn (cross-attention) and out_proj remap.
    """
    new_sd = {}
    consumed = set()

    old_prefixes = set()
    for k in state_dict:
        m = re.match(r"^(.+\.(self_attn|multihead_attn))\.q_proj\.weight$", k)
        if m:
            old_prefixes.add(m.group(1))

    for prefix in old_prefixes:
        q_w = state_dict.get(f"{prefix}.q_proj.weight")
        k_w = state_dict.get(f"{prefix}.k_proj.weight")
        v_w = state_dict.get(f"{prefix}.v_proj.weight")
        q_b = state_dict.get(f"{prefix}.q_proj.bias")
        k_b = state_dict.get(f"{prefix}.k_proj.bias")
        v_b = state_dict.get(f"{prefix}.v_proj.bias")
        o_w = state_dict.get(f"{prefix}.out_proj.weight")
        o_b = state_dict.get(f"{prefix}.out_proj.bias")

        if q_w is not None and k_w is not None and v_w is not None:
            new_sd[f"{prefix}.mha.in_proj_weight"] = torch.cat([q_w, k_w, v_w], dim=0)
            consumed.update([f"{prefix}.q_proj.weight", f"{prefix}.k_proj.weight", f"{prefix}.v_proj.weight"])
            if q_b is not None and k_b is not None and v_b is not None:
                new_sd[f"{prefix}.mha.in_proj_bias"] = torch.cat([q_b, k_b, v_b], dim=0)
                consumed.update([f"{prefix}.q_proj.bias", f"{prefix}.k_proj.bias", f"{prefix}.v_proj.bias"])

        if o_w is not None:
            new_sd[f"{prefix}.mha.out_proj.weight"] = o_w
            consumed.add(f"{prefix}.out_proj.weight")
        if o_b is not None:
            new_sd[f"{prefix}.mha.out_proj.bias"] = o_b
            consumed.add(f"{prefix}.out_proj.bias")

    for k, v in state_dict.items():
        if k not in consumed:
            new_sd[k] = v

    if old_prefixes:
        logger.info("[MHAMigrate] Remapped {} attention modules ({} keys)", len(old_prefixes), len(consumed))

    # Legacy CTC head migration:
    # Some older checkpoints stored a 1-layer linear CTC head as nn.Sequential([nn.Linear])
    # which yields keys like: ctc_head.0.weight / ctc_head.0.bias
    # Current DualHeadModel uses nn.Linear directly: ctc_head.weight / ctc_head.bias
    if "ctc_head.0.weight" in new_sd and "ctc_head.weight" not in new_sd:
        new_sd["ctc_head.weight"] = new_sd.pop("ctc_head.0.weight")
        consumed.add("ctc_head.0.weight")
    if "ctc_head.0.bias" in new_sd and "ctc_head.bias" not in new_sd:
        new_sd["ctc_head.bias"] = new_sd.pop("ctc_head.0.bias")
        consumed.add("ctc_head.0.bias")

    if "ctc_head.weight" in new_sd or "ctc_head.bias" in new_sd:
        # Only log when we actually performed the Sequential->Linear remap.
        if ("ctc_head.0.weight" in state_dict) or ("ctc_head.0.bias" in state_dict):
            logger.info("[CTCMigrate] Remapped legacy ctc_head.0.* -> ctc_head.*")

    return new_sd


def load_model_from_checkpoint(
    ckpt_path: str,
    model: nn.Module,
    device: str = "cpu",
) -> nn.Module:
    """Load checkpoint with automatic legacy MHA key migration."""
    dev = str(device)
    # If the checkpoint was saved on CUDA but this runtime has no CUDA,
    # force-map storages to CPU to avoid deserialization errors.
    if dev.startswith("cuda") and not torch.cuda.is_available():
        map_location = torch.device("cpu")
    else:
        map_location = device

    ckp = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    sd = _migrate_legacy_mha_keys(ckp["model"])
    res = model.load_state_dict(sd, strict=False)
    if res.missing_keys or res.unexpected_keys:
        logger.warning("load_state_dict | missing={} unexpected={}", res.missing_keys, res.unexpected_keys)
    else:
        logger.info("Checkpoint loaded cleanly: {}", ckpt_path)
    # Move to the requested device if it exists; otherwise keep on CPU.
    if dev.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_encoder_features(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    categories: list[str],
    max_samples: int = 2000,
) -> dict:
    """Extract per-frame encoder features with character-level alignment.

    For each sample, the encoder output (B, T_down, D) is aligned to the
    ground-truth label by uniformly distributing characters across frames.

    Returns:
        dict with keys:
            'features':     np.ndarray (N_frames, D)
            'char_labels':  list[str]   per-frame character label
            'char_ids':     np.ndarray (N_frames,) per-frame char index in categories
            'word_labels':  list[str]   per-frame word (the full word this frame belongs to)
            'sample_idx':   np.ndarray (N_frames,) which sample each frame came from
    """
    model.eval()
    model.to(device)

    all_features = []
    all_char_labels = []
    all_char_ids = []
    all_word_labels = []
    all_sample_idx = []
    global_sample = 0

    for x, y, len_x, len_y in dataloader:
        x = x.to(device)

        # Extract encoder features
        if hasattr(model, '_encode_with_mask'):
            # DualHeadModel or BaseModel with AR decoder
            result = model._encode_with_mask(x, len_x.to(device))
            feats = result[0]       # (B, Tm, D)
            enc_pad = result[1]     # (B, Tm) bool
            if len(result) > 2:
                enc_lengths = result[2]
            else:
                enc_lengths = (~enc_pad).sum(dim=1)
        elif hasattr(model, 'encoder'):
            feats = model.encoder(x)  # (B, Tm, D)
            ratio_ds = getattr(model.encoder, 'ratio_ds', 1)
            enc_lengths = (len_x.to(device) // ratio_ds).clamp(min=1, max=feats.size(1))
        else:
            raise ValueError("Model has no encoder or _encode_with_mask method")

        feats_np = feats.cpu().numpy()
        enc_lengths_np = enc_lengths.cpu().numpy()

        B = x.size(0)
        for b in range(B):
            if global_sample >= max_samples:
                break

            T = int(enc_lengths_np[b])
            feat_b = feats_np[b, :T]  # (T, D)

            # Decode label
            L = int(len_y[b])
            lab_ids = y[b, :L].tolist() if y.dim() == 2 else []
            label_str = ''.join(
                categories[i] for i in lab_ids
                if 0 <= i < len(categories) and i != 0  # skip blank
            )

            if len(label_str) == 0 or T == 0:
                continue

            # Uniform character-to-frame alignment
            chars = list(label_str)
            num_chars = len(chars)
            frame_to_char = []
            frame_to_char_id = []
            for t in range(T):
                char_idx = min(int(t * num_chars / T), num_chars - 1)
                c = chars[char_idx]
                frame_to_char.append(c)
                # Find char in categories (skip blank at index 0)
                if c in categories:
                    frame_to_char_id.append(categories.index(c))
                else:
                    frame_to_char_id.append(-1)

            all_features.append(feat_b)
            all_char_labels.extend(frame_to_char)
            all_char_ids.extend(frame_to_char_id)
            all_word_labels.extend([label_str] * T)
            all_sample_idx.extend([global_sample] * T)
            global_sample += 1

        if global_sample >= max_samples:
            break

    logger.info("Extracted features from {} samples, {} total frames", global_sample, len(all_char_labels))

    return {
        'features': np.concatenate(all_features, axis=0),
        'char_labels': all_char_labels,
        'char_ids': np.array(all_char_ids),
        'word_labels': all_word_labels,
        'sample_idx': np.array(all_sample_idx),
    }


@torch.no_grad()
def extract_ctc_posteriors(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    categories: list[str],
    n_samples: int = 10,
) -> list[dict]:
    """Extract CTC posterior heatmaps from the CTC head of a DualHeadModel.

    Returns a list of dicts, each with:
        'posteriors':  np.ndarray (T, V) softmax posteriors
        'label':       str ground-truth label
        'enc_length':  int valid encoder timesteps
    """
    model.eval()
    model.to(device)

    results = []
    count = 0

    for x, y, len_x, len_y in dataloader:
        if count >= n_samples:
            break

        x = x.to(device)
        len_x_dev = len_x.to(device)

        # Need DualHeadModel with CTC head
        if not hasattr(model, 'compute_ctc_logits'):
            raise ValueError("Model does not have compute_ctc_logits (not a DualHeadModel?)")

        feats, enc_pad, enc_lengths = model._encode_with_mask(x, len_x_dev)
        ctc_logits = model.compute_ctc_logits(feats)  # (B, Tm, Vctc)
        ctc_probs = torch.softmax(ctc_logits, dim=-1)  # (B, Tm, Vctc)

        B = x.size(0)
        for b in range(B):
            if count >= n_samples:
                break

            T = int(enc_lengths[b].item())
            probs = ctc_probs[b, :T].cpu().numpy()  # (T, V)

            L = int(len_y[b])
            lab_ids = y[b, :L].tolist() if y.dim() == 2 else []
            label_str = ''.join(
                categories[i] for i in lab_ids
                if 0 <= i < len(categories) and i != 0
            )

            results.append({
                'posteriors': probs,
                'label': label_str,
                'enc_length': T,
            })
            count += 1

    return results
