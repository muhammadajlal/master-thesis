"""Aggregate the just-completed scaled-ungated AR Transformer training on private word
(1709578). Writes results.json + prints 5-fold mean."""
import json
from pathlib import Path
import numpy as np

ROOT = Path("/home/woody/iwso/iwso214h/imu-hwr/results/hwr2/Baseline-AR-Ungated/ar_transformer_s__wi_word_hw6_meta")
MAX_EP = 299


def aggregate(base):
    cers, wers = {}, {}
    fold_dirs = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith("fold_")])
    for i, fd in enumerate(fold_dirs):
        merged = {}
        for fp in sorted(fd.glob("*/train_*.json"), key=lambda p: p.stat().st_mtime):
            try:
                d = json.loads(fp.read_text())
            except Exception:
                continue
            for k, v in d.items():
                if not (isinstance(k, str) and k.isdigit()):
                    continue
                ep = int(k)
                if ep > MAX_EP or not isinstance(v, dict):
                    continue
                ev = v.get("evaluation") or {}
                c = ev.get("character_error_rate")
                w = ev.get("word_error_rate")
                if c is None or w is None:
                    continue
                merged[ep] = (float(c), float(w))
        if not merged:
            continue
        best_ep = min(merged, key=lambda e: merged[e][0])
        c, w = merged[best_ep]
        cers[str(i)] = c
        wers[str(i)] = w
        print(f"  fold_{i}: best-CER epoch {best_ep}, CER {c*100:.2f}%, WER {w*100:.2f}%")
    if not cers:
        return None
    return {
        "cer": {"raw": cers, "mean": float(np.mean(list(cers.values()))), "std": float(np.std(list(cers.values())))},
        "wer": {"raw": wers, "mean": float(np.mean(list(wers.values()))), "std": float(np.std(list(wers.values())))},
    }


print(f"Aggregating: {ROOT}")
r = aggregate(ROOT)
if r:
    (ROOT / "results.json").write_text(json.dumps(r, indent=2))
    print()
    print(f"5-fold CER mean: {r['cer']['mean']*100:.2f}% (std {r['cer']['std']*100:.2f})")
    print(f"5-fold WER mean: {r['wer']['mean']*100:.2f}% (std {r['wer']['std']*100:.2f})")
    print(f"Wrote {ROOT / 'results.json'}")
else:
    print("No data aggregated.")
