#!/usr/bin/env python3
"""Inference MACs for the HWR-GPT (multimodal / VLM) connector variants.

Companion to batch_macs_for_thesis.py (which profiles the classical recognizers).
Same convention as evaluate.py:get_macs_params() so the two are comparable:

  - input  x     = randn(1, num_channel, T), T = 1024 for word datasets
  - profiler     = torch.utils.flop_counter.FlopCounterMode, MACs = FLOPs // 2
  - decode budget L = MEAN_BPE_OUT = 3 GPT-2 subword tokens. This is the rounded-up
    mean OnHW-words500 / private-word output length in the decoder's own units
    (the ~6-character words tokenize to ~2.4 GPT-2 BPE tokens). It is the multimodal
    analogue of the 6-character budget used for the character-level recognizers,
    and reproduces the HWR-GPT MACs reported in the thesis (Chapter 6 tables).
    One teacher-forced forward over L target tokens equals the per-sample compute
    of decoding L tokens with GPT-2 key/value caching, i.e. the real inference cost.
  - auxiliary CTC head is NOT counted (hybrid_mode forced False), except for the
    CTC-posterior connector whose head feeds the connector itself and runs at
    inference.

Output: results/thesis_vlm_macs.json

Usage:
    envs/rewi26/bin/python work/REWI_work/analysis/scripts/vlm_macs_for_thesis.py
"""
import os
import sys
import json

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

import torch
import yaml
from torch.utils.flop_counter import FlopCounterMode
from loguru import logger

logger.remove()  # silence

from rewi.model import build_encoder
from rewi.model.vlm_model import VLMModel

CFG_ROOT = str(PROJ / "configs")

# Rounded-up mean GPT-2 subword output length on the OnHW / private word datasets
# (mean ~2.4 BPE tokens -> 3). The multimodal analogue of the 6-char recognizer
# budget; see the module docstring and the thesis "Inference cost" paragraph.
MEAN_BPE_OUT = 3

VARIANTS: list[tuple[str, str]] = [
    ("Linear",             "G1_minimal_connector/train-G1a-linear-onhw.yaml"),
    ("MLP",                "H1_hybrid_ctc_vlm/train-H1-mlp-onhw.yaml"),
    ("Pool-MLP",           "H1_hybrid_ctc_vlm_pooling/train-H1-pooling-onhw.yaml"),
    ("Conv-Pool",          "G1_minimal_connector/train-G1b-conv-onhw.yaml"),
    ("Q-Former",           "L1_mini_qformer/train-L1-mini-qformer-onhw.yaml"),
    ("Gated Multi-View",   "K5_kv_multiview/train-K5-kv-onhw.yaml"),
    ("KV Slim (L2)",       "L2_kv_slim/train-L2-kv-slim-onhw.yaml"),
    ("CTC-Posterior (K2)", "K2_ctc_posterior/train-K2-lego-onhw.yaml"),
]


def build_vlm(cfgs: dict) -> VLMModel:
    """Mirror evaluate.py:get_macs_params VLM construction exactly."""
    encoder = build_encoder(cfgs["num_channel"], cfgs["arch_en"], cfgs["len_seq"])
    ratio_ds = int(getattr(encoder, "ratio_ds", 1))
    d_cnn = int(cfgs.get("d_cnn", 512))
    vlm_cfg = cfgs.get("vlm", {}) or {}
    lm_name = str(vlm_cfg.get("lm_name", "gpt2"))
    categories = cfgs.get("categories", None)
    vocab_ctc = int(len(categories)) if (categories and bool(vlm_cfg.get("hybrid_ctc", False))) else None

    model = VLMModel(
        encoder=encoder, ratio_ds=ratio_ds, d_cnn=d_cnn, lm_name_or_path=lm_name,
        connector_type=str(vlm_cfg.get("connector_type", "qformer")),
        num_queries=int(vlm_cfg.get("num_queries", 32)),
        qformer_layers=int(vlm_cfg.get("qformer_layers", 4)),
        qformer_nhead=int(vlm_cfg.get("qformer_nhead", 8)),
        qformer_dropout=float(vlm_cfg.get("qformer_dropout", 0.1)),
        prompt_text=vlm_cfg.get("prompt_text", "Transcribe the handwritten text from IMU sensor signals:"),
        refine_prompt_text=vlm_cfg.get("refine_prompt_text", None),
        two_step_decode=bool(vlm_cfg.get("two_step_decode", False)),
        num_soft_tokens=int(vlm_cfg.get("num_soft_tokens", 20)),
        freeze_encoder=bool(cfgs.get("freeze", False)),
        freeze_lm=bool(vlm_cfg.get("freeze_lm", True)),
        use_lora=bool(vlm_cfg.get("use_lora", False)),
        lora_r=int(vlm_cfg.get("lora_r", 16)),
        lora_alpha=int(vlm_cfg.get("lora_alpha", 32)),
        lora_dropout=float(vlm_cfg.get("lora_dropout", 0.05)),
        lora_target_modules=vlm_cfg.get("lora_target_modules", None),
        max_new_tokens=int(vlm_cfg.get("max_new_tokens", 64)),
        num_beams=int(vlm_cfg.get("num_beams", 1)),
        repetition_penalty=float(vlm_cfg.get("repetition_penalty", 1.0)),
        local_files_only=bool(vlm_cfg.get("local_files_only", True)),
        z_dropout=float(vlm_cfg.get("z_dropout", 0.1)),
        label_smoothing=float(vlm_cfg.get("label_smoothing", 0.0)),
        vocab_ctc=vocab_ctc, categories=categories,
    ).eval()
    return model


def profile_variant(name: str, cfg_path: str) -> dict:
    with open(cfg_path) as f:
        cfgs = yaml.safe_load(f)

    torch.manual_seed(42)
    model = build_vlm(cfgs)

    # Bypass the training-only auxiliary CTC branch (not run at inference).
    # For the CTC-posterior connector the head IS the connector input, so it stays.
    aux_ctc_params = 0
    if model.ctc_head is not None:
        aux_ctc_params = sum(p.numel() for p in model.ctc_head.parameters())
    model.hybrid_mode = False

    # Dummy shapes: SAME input convention as evaluate.py; decode budget = mean BPE out.
    T = 1024 if "word" in cfgs["dir_dataset"] else 4096
    x = torch.randn(1, cfgs["num_channel"], T)
    len_x = torch.tensor([T], dtype=torch.long)
    L = MEAN_BPE_OUT
    vocab = int(model.tokenizer.vocab_size)
    labels = torch.randint(0, vocab, (1, L), dtype=torch.long)

    # Probe token counts (outside the FLOP counter)
    with torch.no_grad():
        enc_states, enc_mask = model._encode(x, len_x)
        if model.connector_type in ("ctc_posterior", "lego") and model.ctc_head is not None:
            imu_tokens = model.connector(
                enc_states, enc_mask,
                ctc_logits=model.ctc_head(enc_states),
                lm_embed_weight=model._get_embed_fn().weight,
            )
        else:
            imu_tokens = model.connector(enc_states, enc_mask)
    n_prefix = int(model.prompt_manager.total_prefix_len)  # soft + prompt
    n_imu = int(imu_tokens.size(1))
    n_total = n_prefix + n_imu + L

    flops = 0
    err = None
    try:
        # enable_grad (no backward is called): torch 2.9's FlopCounterMode attaches
        # a ModuleTracker whose multi-grad hooks assert under no_grad when
        # nn.TransformerEncoder takes its fused eval fastpath; enable_grad forces the
        # standard decomposed attention path, which is hook-safe and FLOP-counted.
        with torch.enable_grad():
            with FlopCounterMode(display=False) as fc:
                model(x, len_x, labels)
        flops = int(fc.get_total_flops())
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"

    params_total = sum(p.numel() for p in model.parameters())
    params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return dict(
        variant=name, config=os.path.relpath(cfg_path, CFG_ROOT),
        connector=model.connector_type,
        connector_params=sum(p.numel() for p in model.connector.parameters()),
        params_total=params_total, params_trainable=params_train,
        aux_ctc_params_excluded_from_flops=aux_ctc_params,
        T=T, len_max=L, n_soft=int(model.prompt_manager.num_soft_tokens),
        n_prompt=int(model.prompt_manager.num_prompt_tokens),
        n_imu_tokens=n_imu, n_target=L, n_seq_total=n_total,
        flops=flops, macs=flops // 2, error=err,
    )


def main() -> None:
    out_path = PROJ.parent.parent / "results" / "thesis_vlm_macs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out = []
    print(f"Profiling {len(VARIANTS)} HWR-GPT variants at L={MEAN_BPE_OUT} (mean GPT-2 output)...")
    print(f"{'Variant':20s}  {'Params (M)':>11s}  {'MACs (G)':>10s}")
    print("-" * 48)
    for name, rel in VARIANTS:
        path = os.path.join(CFG_ROOT, rel)
        if not os.path.exists(path):
            print(f"{name:20s}  [NOT FOUND] {rel}")
            out.append(dict(variant=name, config=rel, error="config not found"))
            continue
        try:
            r = profile_variant(name, path)
        except Exception as exc:
            r = dict(variant=name, config=rel, error=f"BUILD FAILED {type(exc).__name__}: {exc}")
        out.append(r)
        if r.get("error"):
            print(f"{name:20s}  [ERROR] {r['error']}")
        else:
            print(f"{name:20s}  {r['params_total']/1e6:11.3f}  {r['macs']/1e9:10.3f}")

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()
