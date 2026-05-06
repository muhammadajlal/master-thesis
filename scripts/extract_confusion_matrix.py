"""Extract per-fold character confusion matrices from baseline AR checkpoints.

For each (dataset, fold) combination, load the AR-Elementwise baseline checkpoint,
run greedy autoregressive decoding on the held-out validation split, align each
prediction to its ground truth via Levenshtein editops, and accumulate
substitution counts into a (vocab_size, vocab_size) matrix. Save as .npy at:

    <out_root>/<dataset>/fold_<k>.npy

The resulting matrix is then loaded at training time by `loops.py` when
`input_corruption.mode = self_confusion`. Per-fold extraction avoids any
cross-fold leakage; the val-set source carries a mild same-fold leak that we
document in the paper.

Usage:
    python scripts/extract_confusion_matrix.py \\
        --dataset wi_word_hw6_meta \\
        --folds 0 1 2 3 4 \\
        --out-root /home/woody/iwso/iwso214h/imu-hwr/results/hwr2/confusion_matrices
"""
import argparse
import os
import sys
import json
import argparse as ap
from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
import Levenshtein

# Allow running from project root
PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from rewi.dataset import HRDataset
from rewi.dataset.utils import fn_collate
from rewi.model import BaseModel
from rewi.utils import seed_everything

# Map dataset → categories list and source baseline-config path. The categories
# match the production baseline YAMLs (word vs sentence vocabulary).
WORD_CATS = ["", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "Ä", "Ö", "Ü", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "ä", "ö", "ü", "ß"]
SENT_CATS = ["", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "ä", "ö", "ü", "Ä", "Ö", "Ü", "ß", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", ",", "(", ")", "'", "?", "!", "+", "=", "-", "/", ";", ":", "·", " "]

EQUATION_CATS = ["", "+", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", "=", "·"]

DATASET_CONFIG = {
    "onhw_wi_word_rh":           {"categories": WORD_CATS},
    "onhw_wd_word_rh":           {"categories": WORD_CATS},
    "wi_word_hw6_meta":          {"categories": WORD_CATS},
    "wi_sent_hw6_meta":          {"categories": SENT_CATS},
    "onhw_equations_wi_word_rh": {"categories": EQUATION_CATS},
    "onhw_equations_wd_word_rh": {"categories": EQUATION_CATS},
}

# Source checkpoint for confusion-matrix extraction. We use the AR-Elementwise
# *vanilla baseline* (no input corruption) so that the confusion matrix reflects
# the errors of a model trained without the technique we are about to apply.
# The WI baselines were saved with a previous attention-module layout (separate
# q_proj/k_proj/v_proj/out_proj instead of nn.MultiheadAttention's combined
# in_proj_weight); a one-shot state_dict translator below repackages those
# weights into the format the current GatedMultiheadAttention wrapper expects.
def _ckpt_template_for(dataset: str) -> str:
    """Pick the right baseline-checkpoint root per dataset.
    WD-split datasets and equation datasets use the freshly-trained AR-Elementwise
    runs (current code path, no state_dict translation needed); the original WI
    Stabilo / OnHW baselines are in the legacy layout (translator handles them).
    """
    if dataset == "onhw_wd_word_rh":
        return "/home/woody/iwso/iwso214h/imu-hwr/results/hwr2/Baseline-AR-ElementwiseGating-WD/ar_transformer_s__{dataset}/fold_{fold}/{fold}/checkpoints/best_cer.pth"
    if dataset.startswith("onhw_equations_wi"):
        return "/home/woody/iwso/iwso214h/imu-hwr/results/hwr2/Baseline-AR-ElementwiseGating-Equations-WI/ar_transformer_s__{dataset}/fold_{fold}/{fold}/checkpoints/best_cer.pth"
    if dataset.startswith("onhw_equations_wd"):
        return "/home/woody/iwso/iwso214h/imu-hwr/results/hwr2/Baseline-AR-ElementwiseGating-Equations-WD/ar_transformer_s__{dataset}/fold_{fold}/{fold}/checkpoints/best_cer.pth"
    # Default: the WI Stabilo / OnHW baselines (legacy state_dict layout).
    return "/home/woody/iwso/iwso214h/imu-hwr/results/hwr2/Baseline-AR-ElementwiseGating/ar_transformer_s__{dataset}/fold_{fold}/{fold}/checkpoints/best_cer.pth"


CKPT_TEMPLATE = "/home/woody/iwso/iwso214h/imu-hwr/results/hwr2/Baseline-AR-ElementwiseGating/ar_transformer_s__{dataset}/fold_{fold}/{fold}/checkpoints/best_cer.pth"


def translate_old_attn_state_dict(state_dict: dict) -> dict:
    """Translate an old-format AR decoder state_dict to the current layout.

    Older checkpoints saved each attention block with separate Q, K, V and
    output projection submodules: ``<prefix>.{q,k,v,out}_proj.{weight,bias}``.
    The current GatedMultiheadAttention wraps an ``nn.MultiheadAttention`` and
    expects the combined Q+K+V projection at ``<prefix>.mha.in_proj_weight``
    (shape ``(3*D, D)``) plus the output projection at
    ``<prefix>.mha.out_proj.{weight,bias}``.

    This function rewrites all matching prefixes in-place by concatenating the
    Q/K/V weights/biases and moving out_proj into the ``mha.`` namespace. Any
    other keys (encoder weights, gating params, projection head, etc.) pass
    through unchanged.
    """
    import re
    translated: dict = {}
    seen_prefixes: set = set()

    # Detect old-format prefixes by looking for `.q_proj.weight` keys.
    pattern = re.compile(r"^(.*)\.q_proj\.weight$")
    for k in state_dict.keys():
        m = pattern.match(k)
        if m:
            seen_prefixes.add(m.group(1))

    for k, v in state_dict.items():
        # Find which (if any) old-format prefix this key belongs to.
        matched_prefix = None
        for prefix in seen_prefixes:
            if k.startswith(prefix + "."):
                matched_prefix = prefix
                break
        if matched_prefix is None:
            translated[k] = v
            continue
        # k is one of: <prefix>.q_proj.{w,b}, k_proj.{w,b}, v_proj.{w,b}, out_proj.{w,b}
        # We process per-prefix once we hit the q_proj.weight key; skip the rest.
        # Suffix paths: q_proj.weight -> mha.in_proj_weight (concat), etc.
        # Simpler: collect everything per prefix in a separate pass below.
        pass

    # Now iterate per old-format prefix and emit the translated entries.
    handled_keys: set = set()
    for prefix in seen_prefixes:
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            for tail in ("weight", "bias"):
                handled_keys.add(f"{prefix}.{proj}.{tail}")
        # Build combined in_proj for Q, K, V.
        for tail in ("weight", "bias"):
            qkv = []
            for proj in ("q_proj", "k_proj", "v_proj"):
                key = f"{prefix}.{proj}.{tail}"
                if key not in state_dict:
                    raise KeyError(
                        f"state_dict missing {key}; cannot build combined in_proj"
                    )
                qkv.append(state_dict[key])
            translated[f"{prefix}.mha.in_proj_{tail}"] = torch.cat(qkv, dim=0)
        # Move out_proj into the mha namespace.
        for tail in ("weight", "bias"):
            old_key = f"{prefix}.out_proj.{tail}"
            new_key = f"{prefix}.mha.out_proj.{tail}"
            if old_key in state_dict:
                translated[new_key] = state_dict[old_key]

    # Pass through anything that wasn't touched by the per-prefix rewrite.
    for k, v in state_dict.items():
        if k in handled_keys:
            continue
        if k not in translated:
            translated[k] = v

    return translated
DATA_TEMPLATE = "/home/woody/iwso/iwso214h/imu-hwr/data/{dataset}"


def build_confusion_matrix_for_fold(dataset: str, fold: int, device: str = "cuda") -> np.ndarray:
    cats = DATASET_CONFIG[dataset]["categories"]
    base = len(cats)
    PAD_ID, BOS_ID, EOS_ID = base, base + 1, base + 2
    num_cls = base + 3
    vocab_size = num_cls

    seed_everything(42)

    # Build model with the canonical AR-Elementwise hyperparameters.
    model = BaseModel(
        arch_en="blconv_b",
        arch_de="ar_transformer_s",
        in_chan=13,
        num_cls=num_cls,
        len_seq=0,
        use_gated_attention=True,
        gating_type="elementwise",
        pad_id=PAD_ID,
    ).to(device)
    ckpt_path = _ckpt_template_for(dataset).format(dataset=dataset, fold=fold)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw_sd = ckpt["model"]
    # Detect the older attention layout by presence of `q_proj.weight` keys.
    needs_translation = any(k.endswith(".q_proj.weight") for k in raw_sd.keys())
    if needs_translation:
        sd = translate_old_attn_state_dict(raw_sd)
    else:
        sd = raw_sd
    model.load_state_dict(sd)
    model.eval()

    # Build val dataloader (same fold).
    dataset_test = HRDataset(
        os.path.join(DATA_TEMPLATE.format(dataset=dataset), "val.json"),
        cats, model.ratio_ds, fold, 0, cache=True,
    )
    dataloader = DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=4, collate_fn=fn_collate)

    confusion = np.zeros((vocab_size, vocab_size), dtype=np.int64)
    n_samples = 0
    n_subs = 0

    with torch.no_grad():
        for batch_idx, (x, y, len_x, len_y) in enumerate(dataloader):
            x = x.to(device)
            B = x.size(0)
            max_len = int(len_y.max().item()) + 2

            # Greedy autoregressive generation.
            y_gen = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
            for _ in range(max_len):
                step_logits = model(x, in_lengths=len_x, y_inp=y_gen)
                nxt = step_logits[:, -1, :].argmax(-1, keepdim=True)
                y_gen = torch.cat([y_gen, nxt], dim=1)

            y_gen_cpu = y_gen.cpu().tolist()
            y_cpu = y.cpu()
            len_y_cpu = len_y.cpu().tolist()

            for b in range(B):
                # Predicted ID sequence: drop BOS, stop at EOS, drop PAD.
                ids_pred: List[int] = []
                for t in y_gen_cpu[b][1:]:
                    if t == EOS_ID:
                        break
                    if t == PAD_ID:
                        continue
                    ids_pred.append(int(t))

                # Ground-truth ID sequence (right-padded).
                if y.dim() == 2:
                    L = int(len_y_cpu[b])
                    ids_true = [int(v) for v in y_cpu[b, :L].tolist()]
                else:
                    raise RuntimeError("expected 2D padded label tensor")

                # Use Levenshtein.editops on string-encoded IDs (works on lists too via Latin-1 trick).
                # We need substitution operations: pairs of (true_pos, pred_pos) where character differs.
                # Levenshtein.editops takes strings; encode IDs as single Latin-1 chars via chr().
                # IDs may exceed 255 for sentence vocab? num_cls(sent) = 85 + 3 = 88, fine.
                if vocab_size > 256:
                    raise RuntimeError("vocab > 256 not supported by chr() trick; use a custom DP")
                s_true = "".join(chr(i) for i in ids_true)
                s_pred = "".join(chr(i) for i in ids_pred)
                ops = Levenshtein.editops(s_true, s_pred)
                for op, ti, pi in ops:
                    if op == "replace":
                        true_id = ids_true[ti]
                        pred_id = ids_pred[pi]
                        # Skip specials and CTC-blank entirely.
                        if true_id in (0, PAD_ID, BOS_ID, EOS_ID): continue
                        if pred_id in (0, PAD_ID, BOS_ID, EOS_ID): continue
                        confusion[true_id, pred_id] += 1
                        n_subs += 1
                n_samples += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"  batch {batch_idx+1}/{len(dataloader)}  samples={n_samples}  subs={n_subs}", flush=True)

    return confusion, n_samples, n_subs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASET_CONFIG.keys()))
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--out-root", required=True, help="output root for confusion matrices")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out_dir = Path(args.out_root) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    for fold in args.folds:
        out_path = out_dir / f"fold_{fold}.npy"
        if out_path.exists():
            print(f"[{args.dataset}][fold={fold}] exists, skipping {out_path}")
            continue
        print(f"[{args.dataset}][fold={fold}] extracting…", flush=True)
        conf, n_samples, n_subs = build_confusion_matrix_for_fold(args.dataset, fold, args.device)
        np.save(out_path, conf)
        diag = int(conf.diagonal().sum())
        total = int(conf.sum())
        print(f"[{args.dataset}][fold={fold}] done. samples={n_samples} subs={n_subs} "
              f"matrix_total={total} (diag={diag}, off-diag={total - diag}) → {out_path}")


if __name__ == "__main__":
    main()
