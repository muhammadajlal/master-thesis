#!/usr/bin/env python3
"""Visualize VLM embedding spaces using PCA/UMAP.

Extracts embeddings at two points:
  V1: Connector output (IMU tokens after Q-Former/MLP projection)
  V2: GPT-2 decoder last hidden state

Produces scatter plots colored by dataset (public vs private) and
optionally by character/word label.

Usage:
    python scripts/visualize_embeddings.py \
        --ckpt_onhw results/hwr2/vlm_Qformer_gpt2_word_v2/vlm__onhw_wi_word_rh/fold_0/0/checkpoints/best_cer.pth \
        --ckpt_hw6 results/hwr2/vlm_Qformer_gpt2_word_v2/vlm__wi_word_hw6_meta/fold_0/0/checkpoints/best_cer.pth \
        --config_onhw configs/vlm_ablation/train-vlm-A1-mlp-pretrained.yaml \
        --config_hw6 configs/vlm_ablation/train-vlm-A1-mlp-pretrained.yaml \
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
        # out: (B, K, d_lm) — IMU tokens
        self.connector_out.append(out.detach().cpu())

    def _decoder_hook(self, module, inp, out):
        # GPT-2 layer output: tuple of (hidden_states, ...)
        if isinstance(out, tuple):
            hidden = out[0]
        else:
            hidden = out
        self.decoder_out.append(hidden.detach().cpu())

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

def load_dataset(config_path: str, fold: int = 0, split: str = "test"):
    """Load dataset from config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dir_dataset = cfg["dir_dataset"]
    categories = cfg["categories"]

    ds = HRDataset(
        dir_dataset,
        fold=fold,
        split=split,
        in_chan=cfg["num_channel"],
        categories=categories,
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
    """Extract connector and decoder embeddings from the model."""
    collector = EmbeddingCollector()
    collector.register(model)

    labels_all = []
    n_collected = 0

    for batch in dataloader:
        if n_collected >= max_samples:
            break

        x = batch["x"].to(device)
        len_x = batch["len_x"].to(device)
        lm_labels = batch["lm_labels"].to(device)
        texts = batch.get("texts", [])

        # Forward pass to trigger hooks
        _ = model(x, len_x, lm_labels, texts=texts)

        labels_all.extend(texts if texts else [""] * x.size(0))
        n_collected += x.size(0)

    collector.remove()

    # Aggregate: mean-pool across sequence dim to get one vector per sample
    connector_embs = torch.cat(collector.connector_out, dim=0)[:max_samples]
    decoder_embs = torch.cat(collector.decoder_out, dim=0)[:max_samples]

    # Mean-pool: (N, K, d) → (N, d)
    connector_pooled = connector_embs.mean(dim=1).numpy()
    decoder_pooled = decoder_embs.mean(dim=1).numpy()

    return {
        "connector": connector_pooled,
        "decoder": decoder_pooled,
        "labels": labels_all[:max_samples],
    }


# ── Dimensionality reduction + plotting ────────────────────

def reduce_and_plot(
    embs_onhw: np.ndarray,
    embs_hw6: np.ndarray,
    method: str,
    title: str,
    out_path: str,
):
    """Apply PCA or UMAP and create scatter plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    combined = np.concatenate([embs_onhw, embs_hw6], axis=0)
    n_onhw = len(embs_onhw)

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
            return
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
        coords = reducer.fit_transform(combined)
        axis_labels = ("UMAP-1", "UMAP-2")
    else:
        raise ValueError(f"Unknown method: {method}")

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


# ── Main ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize VLM embedding spaces")
    parser.add_argument("--ckpt_onhw", required=True, help="VLM checkpoint for OnHW dataset")
    parser.add_argument("--ckpt_hw6", required=True, help="VLM checkpoint for HW6 dataset")
    parser.add_argument("--config_onhw", required=True, help="Config YAML for OnHW")
    parser.add_argument("--config_hw6", required=True, help="Config YAML for HW6")
    parser.add_argument("--out_dir", default="analysis/embedding_viz", help="Output directory")
    parser.add_argument("--method", default="pca", choices=["pca", "umap"], help="Reduction method")
    parser.add_argument("--max_samples", type=int, default=500, help="Max samples per dataset")
    parser.add_argument("--fold", type=int, default=0, help="Fold to use")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for extraction")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    # Load models
    logger.info("Loading OnHW model...")
    model_onhw = load_vlm_from_checkpoint(args.config_onhw, args.ckpt_onhw, args.device)

    logger.info("Loading HW6 model...")
    model_hw6 = load_vlm_from_checkpoint(args.config_hw6, args.ckpt_hw6, args.device)

    # Load datasets
    logger.info("Loading OnHW dataset (fold {})...", args.fold)
    ds_onhw, cats_onhw = load_dataset(args.config_onhw, fold=args.fold, split="test")

    logger.info("Loading HW6 dataset (fold {})...", args.fold)
    ds_hw6, cats_hw6 = load_dataset(args.config_hw6, fold=args.fold, split="test")

    tokenizer = model_onhw.tokenizer
    collate_fn = lambda batch: vlm_collate(batch, tokenizer, max_label_len=96)

    dl_onhw = DataLoader(ds_onhw, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    dl_hw6 = DataLoader(ds_hw6, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # Extract embeddings
    logger.info("Extracting OnHW embeddings...")
    embs_onhw = extract_embeddings(model_onhw, dl_onhw, args.device, args.max_samples)

    logger.info("Extracting HW6 embeddings...")
    embs_hw6 = extract_embeddings(model_hw6, dl_hw6, args.device, args.max_samples)

    # Save raw embeddings
    np.savez(
        str(out_dir / f"embeddings_fold{args.fold}.npz"),
        connector_onhw=embs_onhw["connector"],
        connector_hw6=embs_hw6["connector"],
        decoder_onhw=embs_onhw["decoder"],
        decoder_hw6=embs_hw6["decoder"],
        labels_onhw=embs_onhw["labels"],
        labels_hw6=embs_hw6["labels"],
    )
    logger.info("Saved raw embeddings to {}", out_dir / f"embeddings_fold{args.fold}.npz")

    # V1: Connector output visualization
    reduce_and_plot(
        embs_onhw["connector"],
        embs_hw6["connector"],
        method=args.method,
        title=f"V1: Connector Output ({args.method.upper()}) — Fold {args.fold}",
        out_path=str(out_dir / f"V1_connector_{args.method}_fold{args.fold}.png"),
    )

    # V2: Decoder hidden state visualization
    reduce_and_plot(
        embs_onhw["decoder"],
        embs_hw6["decoder"],
        method=args.method,
        title=f"V2: Decoder Last Hidden State ({args.method.upper()}) — Fold {args.fold}",
        out_path=str(out_dir / f"V2_decoder_{args.method}_fold{args.fold}.png"),
    )

    logger.info("Done! Plots saved to {}", out_dir)


if __name__ == "__main__":
    main()
