#!/usr/bin/env python3
"""Visualize VLM embedding spaces using PCA/UMAP.

Produces three types of visualizations:

  V1: Connector output — colored by dataset (OnHW vs HW6)
      Shows whether public/private data occupy the same region after projection.

  V2: Decoder last hidden state — colored by dataset (OnHW vs HW6)
      Shows whether GPT-2 processes both datasets similarly.

  V3: Modality alignment — IMU tokens vs text anchors in the SAME space
      Shows the modality gap: are projected IMU features close to the
      ground-truth text embeddings in GPT-2's input embedding space?
      Small dots = IMU tokens (sensor), large circles = text anchors (GT label).

Usage:
    python scripts/visualize_embeddings.py \
        --ckpt_onhw results/hwr2/.../best_cer.pth \
        --ckpt_hw6 results/hwr2/.../best_cer.pth \
        --config_onhw configs/... \
        --config_hw6 configs/... \
        --out_dir analysis/embedding_viz \
        --method pca \
        --max_samples 500
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from loguru import logger

# Add project root to path
PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from rewi.dataset import HRDataset
from rewi.dataset.lm_collate import vlm_collate
from rewi.dataset.utils import fn_collate
from rewi.model import build_encoder
from rewi.model.vlm_model import VLMModel
from torch.utils.data import DataLoader


# ── Embedding extraction via hooks ─────────────────────────

class EmbeddingCollector:
    """Collect embeddings from VLM model via forward hooks."""

    def __init__(self):
        self.connector_out: list[torch.Tensor] = []
        self.decoder_out: list[torch.Tensor] = []
        self._handles = []

    def register(self, model: VLMModel):
        """Register hooks on connector and LM transformer."""
        # Hook 1: connector output (IMU tokens)
        h1 = model.connector.register_forward_hook(self._connector_hook)
        self._handles.append(h1)

        # Hook 2: GPT-2 last transformer layer output
        lm = model.lm
        if hasattr(lm, "get_base_model"):
            lm = lm.get_base_model()
        if hasattr(lm, "transformer"):
            # GPT-2: transformer.h[-1] is the last layer
            last_layer = lm.transformer.h[-1]
        elif hasattr(lm, "decoder"):
            # T5-like: decoder.block[-1]
            last_layer = lm.decoder.block[-1]
        else:
            logger.warning("Cannot find last transformer layer, skipping decoder hook")
            return

        h2 = last_layer.register_forward_hook(self._decoder_hook)
        self._handles.append(h2)

    def _connector_hook(self, module, inp, out):
        # out: (B, K, d_lm) — IMU tokens; mean-pool over K
        self.connector_out.append(out.detach().cpu().mean(dim=1))  # (B, d_lm)

    def _decoder_hook(self, module, inp, out):
        # GPT-2 layer output: tuple of (hidden_states, ...)
        if isinstance(out, tuple):
            hidden = out[0]
        else:
            hidden = out
        # Mean-pool over variable seq_len to get (B, d_lm)
        self.decoder_out.append(hidden.detach().cpu().mean(dim=1))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ── Model loading ──────────────────────────────────────────

def load_vlm_from_checkpoint(config_path: str, ckpt_path: str, device: str = "cpu") -> VLMModel:
    """Load VLM model from config + checkpoint."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    vlm_cfg = cfg.get("vlm", {})
    encoder = build_encoder(cfg["num_channel"], cfg["arch_en"], cfg.get("len_seq", 0))

    model = VLMModel(
        encoder=encoder,
        ratio_ds=encoder.ratio_ds,
        d_cnn=cfg["d_cnn"],
        lm_name_or_path=vlm_cfg["lm_name"],
        connector_type=vlm_cfg.get("connector_type", "qformer"),
        num_queries=vlm_cfg.get("num_queries", 32),
        qformer_layers=vlm_cfg.get("qformer_layers", 4),
        qformer_nhead=vlm_cfg.get("qformer_nhead", 8),
        qformer_dropout=vlm_cfg.get("qformer_dropout", 0.1),
        prompt_text=vlm_cfg.get("prompt_text", ""),
        refine_prompt_text=vlm_cfg.get("refine_prompt_text", None),
        two_step_decode=bool(vlm_cfg.get("two_step_decode", False)),
        num_soft_tokens=vlm_cfg.get("num_soft_tokens", 20),
        freeze_encoder=cfg.get("freeze", False),
        freeze_lm=vlm_cfg.get("freeze_lm", True),
        use_lora=vlm_cfg.get("use_lora", False),
        lora_r=vlm_cfg.get("lora_r", 16),
        lora_alpha=vlm_cfg.get("lora_alpha", 32),
        lora_dropout=vlm_cfg.get("lora_dropout", 0.05),
        lora_target_modules=vlm_cfg.get("lora_target_modules", None),
        max_new_tokens=vlm_cfg.get("max_new_tokens", 64),
        num_beams=vlm_cfg.get("num_beams", 1),
        local_files_only=vlm_cfg.get("local_files_only", True),
        z_dropout=0.0,  # No dropout at inference
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"]

    # Handle legacy qformer→connector key migration
    if any(k.startswith("qformer.") for k in sd):
        sd = {
            (f"connector.{k[len('qformer.'):]}" if k.startswith("qformer.") else k): v
            for k, v in sd.items()
        }

    res = model.load_state_dict(sd, strict=False)
    logger.info("Loaded checkpoint: missing={}, unexpected={}", len(res.missing_keys), len(res.unexpected_keys))

    model = model.to(device).eval()
    return model


# ── Data loading ───────────────────────────────────────────

def load_dataset(config_path: str, fold: int = 0):
    """Load dataset from config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dir_dataset = cfg["dir_dataset"]
    categories = cfg["categories"]
    path_anno = os.path.join(dir_dataset, "val.json")

    ds = HRDataset(
        path_anno=path_anno,
        categories=categories,
        ratio_ds=cfg.get("ratio_ds", 4),
        idx_fold=fold,
        cache=False,
        aug=False,
    )
    return ds, categories


# ── Embedding extraction ───────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    model: VLMModel,
    dataloader: DataLoader,
    device: str,
    max_samples: int = 500,
) -> dict:
    """Extract connector, decoder, and text anchor embeddings.

    Returns dict with keys:
        connector:   (N, d_lm) — mean-pooled connector output (IMU tokens)
        decoder:     (N, d_lm) — mean-pooled decoder last hidden state
        text_anchor: (N, d_lm) — mean-pooled GT text embeddings from LM table
        labels:      list[str] — ground-truth text strings
    """
    collector = EmbeddingCollector()
    collector.register(model)

    # Get the LM embedding function for text anchors
    embed_fn = model._get_embed_fn()

    labels_all = []
    text_anchor_list = []
    n_collected = 0

    for batch in dataloader:
        if n_collected >= max_samples:
            break

        # vlm_collate returns (x, len_x, labels, texts)
        x, len_x, lm_labels, texts = batch[0], batch[1], batch[2], batch[3]
        x = x.to(device)
        len_x = len_x.to(device)
        lm_labels = lm_labels.to(device)

        # Forward pass to trigger hooks
        _ = model(x, len_x, lm_labels, texts=texts)

        # Extract text anchor embeddings from the LM embedding table
        # lm_labels: (B, L) with -100 for padding
        text_ids = lm_labels.clone()
        pad_mask = text_ids == -100
        text_ids[pad_mask] = model.tokenizer.pad_token_id
        text_emb = embed_fn(text_ids)  # (B, L, d_lm)

        # Mean-pool over non-padding positions per sample
        # mask: (B, L, 1) — 1 for real tokens, 0 for padding
        valid_mask = (~pad_mask).unsqueeze(-1).float().to(device)  # (B, L, 1)
        text_sum = (text_emb * valid_mask).sum(dim=1)  # (B, d_lm)
        text_count = valid_mask.sum(dim=1).clamp(min=1)  # (B, 1)
        text_pooled = (text_sum / text_count).detach().cpu()  # (B, d_lm)
        text_anchor_list.append(text_pooled)

        labels_all.extend(texts if texts else [""] * x.size(0))
        n_collected += x.size(0)

    collector.remove()

    # Aggregate — already mean-pooled to (B, d_lm) per batch
    connector_pooled = torch.cat(collector.connector_out, dim=0)[:max_samples].numpy()
    decoder_pooled = torch.cat(collector.decoder_out, dim=0)[:max_samples].numpy()
    text_anchors = torch.cat(text_anchor_list, dim=0)[:max_samples].numpy()

    return {
        "connector": connector_pooled,
        "decoder": decoder_pooled,
        "text_anchor": text_anchors,
        "labels": labels_all[:max_samples],
    }


# ── Dimensionality reduction helpers ──────────────────────

def _fit_reducer(combined: np.ndarray, method: str):
    """Fit a reducer and return (coords, axis_labels)."""
    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(combined)
        var_explained = reducer.explained_variance_ratio_
        axis_labels = (
            f"PC1 ({var_explained[0]:.1%})",
            f"PC2 ({var_explained[1]:.1%})",
        )
    elif method == "umap":
        try:
            import umap
        except ImportError:
            logger.error("UMAP not installed. Install with: pip install umap-learn")
            return None, None
        reducer = umap.UMAP(
            n_components=2, random_state=42, n_neighbors=30, min_dist=0.3,
        )
        coords = reducer.fit_transform(combined)
        axis_labels = ("UMAP-1", "UMAP-2")
    else:
        raise ValueError(f"Unknown method: {method}")
    return coords, axis_labels


# ── V1/V2: Dataset comparison plots ──────────────────────

def plot_dataset_comparison(
    embs_onhw: np.ndarray,
    embs_hw6: np.ndarray,
    method: str,
    title: str,
    out_path: str,
):
    """V1/V2: scatter plot colored by dataset (OnHW vs HW6)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    combined = np.concatenate([embs_onhw, embs_hw6], axis=0)
    n_onhw = len(embs_onhw)

    coords, axis_labels = _fit_reducer(combined, method)
    if coords is None:
        return

    coords_onhw = coords[:n_onhw]
    coords_hw6 = coords[n_onhw:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        coords_onhw[:, 0], coords_onhw[:, 1],
        c="tab:blue", alpha=0.5, s=15, label="OnHW (public)",
    )
    ax.scatter(
        coords_hw6[:, 0], coords_hw6[:, 1],
        c="tab:orange", alpha=0.5, s=15, label="HW6 (private)",
    )
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: {}", out_path)


# ── V3: Modality alignment plot ──────────────────────────

def plot_modality_alignment(
    imu_embs: np.ndarray,
    text_embs: np.ndarray,
    method: str,
    title: str,
    out_path: str,
    dataset_label: str = "",
):
    """V3: IMU sensor tokens vs GT text anchors in the same embedding space.

    Shows the modality gap: are projected IMU features close to the
    ground-truth text embeddings in GPT-2's input embedding space?

    Args:
        imu_embs:      (N, d) mean-pooled connector output (sensor modality)
        text_embs:     (N, d) mean-pooled GT text embeddings (text modality)
        method:        'pca' or 'umap'
        title:         Plot title
        out_path:      Output file path
        dataset_label: e.g. "OnHW" or "HW6" for legend labels
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    combined = np.concatenate([imu_embs, text_embs], axis=0)
    n_imu = len(imu_embs)

    coords, axis_labels = _fit_reducer(combined, method)
    if coords is None:
        return

    coords_imu = coords[:n_imu]
    coords_text = coords[n_imu:]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Text anchors: large transparent circles (background)
    ax.scatter(
        coords_text[:, 0], coords_text[:, 1],
        c="tab:green", alpha=0.25, s=120, marker="o",
        label=f"Text anchors (GT labels)",
        edgecolors="tab:green", linewidths=0.5,
    )
    # IMU tokens: small solid dots (foreground)
    ax.scatter(
        coords_imu[:, 0], coords_imu[:, 1],
        c="tab:red", alpha=0.6, s=15,
        label=f"IMU tokens (sensor)",
    )

    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_title(title)
    ax.legend(fontsize=11)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: {}", out_path)


def plot_modality_alignment_both_datasets(
    imu_onhw: np.ndarray, text_onhw: np.ndarray,
    imu_hw6: np.ndarray, text_hw6: np.ndarray,
    method: str, title: str, out_path: str,
):
    """V3 combined: both datasets in one plot with 4 groups.

    Shows IMU tokens and text anchors for both OnHW and HW6 together,
    revealing both the modality gap and the dataset distribution gap.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    combined = np.concatenate(
        [imu_onhw, text_onhw, imu_hw6, text_hw6], axis=0,
    )
    n1 = len(imu_onhw)
    n2 = n1 + len(text_onhw)
    n3 = n2 + len(imu_hw6)

    coords, axis_labels = _fit_reducer(combined, method)
    if coords is None:
        return

    c_imu_onhw = coords[:n1]
    c_text_onhw = coords[n1:n2]
    c_imu_hw6 = coords[n2:n3]
    c_text_hw6 = coords[n3:]

    fig, ax = plt.subplots(figsize=(12, 9))

    # Text anchors (large, transparent)
    ax.scatter(
        c_text_onhw[:, 0], c_text_onhw[:, 1],
        c="tab:blue", alpha=0.2, s=120, marker="o",
        label="Text anchors — OnHW",
        edgecolors="tab:blue", linewidths=0.5,
    )
    ax.scatter(
        c_text_hw6[:, 0], c_text_hw6[:, 1],
        c="tab:orange", alpha=0.2, s=120, marker="o",
        label="Text anchors — HW6",
        edgecolors="tab:orange", linewidths=0.5,
    )

    # IMU tokens (small, solid)
    ax.scatter(
        c_imu_onhw[:, 0], c_imu_onhw[:, 1],
        c="tab:blue", alpha=0.7, s=15, marker="^",
        label="IMU tokens — OnHW",
    )
    ax.scatter(
        c_imu_hw6[:, 0], c_imu_hw6[:, 1],
        c="tab:orange", alpha=0.7, s=15, marker="^",
        label="IMU tokens — HW6",
    )

    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_title(title)
    ax.legend(fontsize=10)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: {}", out_path)


# ── Main ───────────────────────────────────────────────────

def _load_and_extract(config_path, ckpt_path, fold, batch_size, device, max_samples, label):
    """Load model + dataset and extract embeddings. Returns None on failure."""
    try:
        logger.info("Loading {} model from {}...", label, ckpt_path)
        model = load_vlm_from_checkpoint(config_path, ckpt_path, device)

        logger.info("Loading {} dataset (fold {})...", label, fold)
        ds, _ = load_dataset(config_path, fold=fold)

        with open(config_path) as _f:
            _cfg = yaml.safe_load(_f)
        categories = _cfg["categories"]
        hf_tok = model.tokenizer
        pad_id = 0  # CTC blank / PAD
        collate_fn = lambda batch: vlm_collate(
            batch,
            base_collate_fn=fn_collate,
            hf_tokenizer=hf_tok,
            categories=categories,
            pad_id=pad_id,
            max_label_len=int(_cfg.get("lm_max_label_len", 96)),
        )
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

        logger.info("Extracting {} embeddings...", label)
        embs = extract_embeddings(model, dl, device, max_samples)

        # Free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return embs
    except Exception as e:
        logger.error("Failed to load/extract {}: {}", label, e)
        return None


def main():
    parser = argparse.ArgumentParser(description="Visualize VLM embedding spaces")
    parser.add_argument("--ckpt_onhw", default=None, help="VLM checkpoint for OnHW dataset")
    parser.add_argument("--ckpt_hw6", default=None, help="VLM checkpoint for HW6 dataset")
    parser.add_argument("--config_onhw", default=None, help="Config YAML for OnHW")
    parser.add_argument("--config_hw6", default=None, help="Config YAML for HW6")
    parser.add_argument("--out_dir", default="analysis/embedding_viz", help="Output directory")
    parser.add_argument("--method", default="pca", choices=["pca", "umap"], help="Reduction method")
    parser.add_argument("--max_samples", type=int, default=500, help="Max samples per dataset")
    parser.add_argument("--fold", type=int, default=0, help="Fold to use")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for extraction")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    has_onhw = args.ckpt_onhw and args.config_onhw
    has_hw6 = args.ckpt_hw6 and args.config_hw6

    if not has_onhw and not has_hw6:
        logger.error("Provide at least one of --ckpt_onhw/--config_onhw or --ckpt_hw6/--config_hw6")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    m = args.method
    f = args.fold

    embs_onhw = None
    embs_hw6 = None

    if has_onhw:
        embs_onhw = _load_and_extract(
            args.config_onhw, args.ckpt_onhw, args.fold,
            args.batch_size, args.device, args.max_samples, "OnHW",
        )

    if has_hw6:
        embs_hw6 = _load_and_extract(
            args.config_hw6, args.ckpt_hw6, args.fold,
            args.batch_size, args.device, args.max_samples, "HW6",
        )

    # Save raw embeddings
    save_dict = {}
    if embs_onhw:
        save_dict.update(
            connector_onhw=embs_onhw["connector"],
            decoder_onhw=embs_onhw["decoder"],
            text_anchor_onhw=embs_onhw["text_anchor"],
            labels_onhw=embs_onhw["labels"],
        )
    if embs_hw6:
        save_dict.update(
            connector_hw6=embs_hw6["connector"],
            decoder_hw6=embs_hw6["decoder"],
            text_anchor_hw6=embs_hw6["text_anchor"],
            labels_hw6=embs_hw6["labels"],
        )
    if save_dict:
        os.makedirs(str(out_dir), exist_ok=True)
        np.savez(str(out_dir / f"embeddings_fold{f}.npz"), **save_dict)
        logger.info("Saved raw embeddings to {}", out_dir / f"embeddings_fold{f}.npz")

    # ── V1/V2: Dataset comparison (requires both) ──────────
    if embs_onhw and embs_hw6:
        plot_dataset_comparison(
            embs_onhw["connector"],
            embs_hw6["connector"],
            method=m,
            title=f"V1: Connector Output ({m.upper()}) — Fold {f}",
            out_path=str(out_dir / f"V1_connector_{m}_fold{f}.png"),
        )

        plot_dataset_comparison(
            embs_onhw["decoder"],
            embs_hw6["decoder"],
            method=m,
            title=f"V2: Decoder Last Hidden State ({m.upper()}) — Fold {f}",
            out_path=str(out_dir / f"V2_decoder_{m}_fold{f}.png"),
        )

    # ── V3: Modality alignment — per dataset ───────────────
    if embs_onhw:
        plot_modality_alignment(
            embs_onhw["connector"],
            embs_onhw["text_anchor"],
            method=m,
            title=f"V3a: Modality Gap — OnHW ({m.upper()}) — Fold {f}",
            out_path=str(out_dir / f"V3a_modality_onhw_{m}_fold{f}.png"),
            dataset_label="OnHW",
        )

    if embs_hw6:
        plot_modality_alignment(
            embs_hw6["connector"],
            embs_hw6["text_anchor"],
            method=m,
            title=f"V3b: Modality Gap — HW6 ({m.upper()}) — Fold {f}",
            out_path=str(out_dir / f"V3b_modality_hw6_{m}_fold{f}.png"),
            dataset_label="HW6",
        )

    # ── V3c: Modality alignment — both datasets combined ───
    if embs_onhw and embs_hw6:
        plot_modality_alignment_both_datasets(
            embs_onhw["connector"], embs_onhw["text_anchor"],
            embs_hw6["connector"], embs_hw6["text_anchor"],
            method=m,
            title=f"V3c: Modality Gap — Both Datasets ({m.upper()}) — Fold {f}",
            out_path=str(out_dir / f"V3c_modality_both_{m}_fold{f}.png"),
        )

    logger.info("Done! All plots saved to {}", out_dir)


if __name__ == "__main__":
    main()
