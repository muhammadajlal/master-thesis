#!/usr/bin/env python3
"""Clone the 23 in-scope Chapter 6 multimodal configs from lambda_ctc=0.6
to lambda_ctc=0.2 with distinct dir_work, and emit a manifest the
training sbatch reads.

The 23 configs are exactly the ones cited in vlm_results.tex tables:
  - tab:vlm-h1            (H1_mlp on private word; onhw already in H1 sweep)
  - tab:vlm-pooling-series (H1/H2/H3_pooling on onhw+word)
  - tab:contrastive-results (J1_mlp/J1_pooling on onhw+word)
  - tab:vlm-phase6         (K1/K2/K3/K4 on onhw+word)
  - tab:vlm-l-series       (L1/L2 on onhw+word)

Output:
  configs/_lam02_ch6/<group>/<basename>_lam02.yaml   (23 files)
  scripts/_lam02_ch6_manifest.txt                    (one path per line)
"""
from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent  # work/REWI_work
CFG_ROOT = ROOT / "configs"
OUT_CFG_ROOT = CFG_ROOT / "_lam02_ch6"
OUT_MANIFEST = ROOT / "scripts" / "_lam02_ch6_manifest.txt"

# (source_relative_path_from_configs)
SOURCES = [
    # tab:vlm-h1 — onhw already covered by H1 lambda sweep; private word missing
    "H1_hybrid_ctc_vlm/train-H1-mlp-word.yaml",
    # tab:vlm-pooling-series
    "H1_hybrid_ctc_vlm_pooling/train-H1-pooling-onhw.yaml",
    "H1_hybrid_ctc_vlm_pooling/train-H1-pooling-word.yaml",
    "H2_hybrid_prompt_augmentation_pooling/train-H2-pooling-onhw.yaml",
    "H2_hybrid_prompt_augmentation_pooling/train-H2-pooling-word.yaml",
    "H3_hybrid_two_step_prompt_pooling/train-H3-pooling-onhw.yaml",
    "H3_hybrid_two_step_prompt_pooling/train-H3-pooling-word.yaml",
    # tab:contrastive-results / -alignment.
    # NOTE: J1 and J2 have byte-identical configs but ran on DIFFERENT code. The J1 base
    # runs (2026-03-22/23, commit dc8c03b) predate the contrastive logit-scale fix
    # (log_tau = log(0.07), i.e. tau = 0.07 instead of 1/0.07 = 14.29) and are superseded;
    # J2 (2026-03-24/25) used the corrected working-tree implementation, subsequently
    # committed as 78abeb3 on 2026-03-27, and supplies the reported lambda = 0.6 results.
    # These clones are lambda = 0.2 and ran 2026-06-28, long after the fix, so they use the
    # corrected loss despite the J1_ prefix. See REPRODUCIBILITY.md, "Superseded contrastive runs".
    "J1_contrastive_mlp/train-J1-mlp-onhw.yaml",
    "J1_contrastive_mlp/train-J1-mlp-word.yaml",
    "J1_contrastive_pooling/train-J1-pooling-onhw.yaml",
    "J1_contrastive_pooling/train-J1-pooling-word.yaml",
    # tab:vlm-phase6
    "K1_ctc_mse/train-K1-mlp-onhw.yaml",
    "K1_ctc_mse/train-K1-mlp-word.yaml",
    "K2_ctc_posterior/train-K2-lego-onhw.yaml",
    "K2_ctc_posterior/train-K2-lego-word.yaml",
    "K3_ec_loss/train-K3-ec-onhw.yaml",
    "K3_ec_loss/train-K3-ec-word.yaml",
    "K4_sea_contrastive/train-K4-sea-onhw.yaml",
    "K4_sea_contrastive/train-K4-sea-word.yaml",
    # tab:vlm-l-series (param-matched 5.1M connectors)
    "L1_mini_qformer/train-L1-mini-qformer-onhw.yaml",
    "L1_mini_qformer/train-L1-mini-qformer-word.yaml",
    "L2_kv_slim/train-L2-kv-slim-onhw.yaml",
    "L2_kv_slim/train-L2-kv-slim-word.yaml",
]

LAM_RE = re.compile(r"^(\s*hybrid_lambda_ctc:\s*)0\.6(\s*)$", flags=re.MULTILINE)
DIRWORK_RE = re.compile(r"^(dir_work:\s*)(\S+)$", flags=re.MULTILINE)


def patch(src: str) -> str:
    out, n = LAM_RE.subn(r"\g<1>0.2\g<2>", src, count=1)
    if n != 1:
        raise RuntimeError("expected exactly one hybrid_lambda_ctc: 0.6 occurrence")
    out2, m = DIRWORK_RE.subn(lambda mo: f"{mo.group(1)}{mo.group(2)}_lam02", out, count=1)
    if m != 1:
        raise RuntimeError("expected exactly one dir_work line")
    return out2


def main() -> None:
    OUT_CFG_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_lines = []
    for rel in SOURCES:
        src_path = CFG_ROOT / rel
        if not src_path.exists():
            raise FileNotFoundError(src_path)
        src_text = src_path.read_text()
        dst_text = patch(src_text)
        # Group-keyed subdir keeps clones grouped
        group = Path(rel).parent.name
        base = Path(rel).stem + "_lam02.yaml"
        dst_path = OUT_CFG_ROOT / group / base
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_text(dst_text)
        manifest_lines.append(str(dst_path.relative_to(ROOT)))
        print(f"cloned: {rel} -> {dst_path.relative_to(ROOT)}")

    OUT_MANIFEST.write_text("\n".join(manifest_lines) + "\n")
    print(f"\nwrote manifest ({len(manifest_lines)} configs): {OUT_MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
