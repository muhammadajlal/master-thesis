#!/usr/bin/env python3
"""Clone the L1 mini Q-Former and L2 KV-slim base configs (hybrid CTC at
lambda=0.6) to no-CTC variants for the Chapter 6 fair-comparison sweep.

Each clone keeps everything from the source EXCEPT:
  - vlm.hybrid_ctc: true  ->  vlm.hybrid_ctc: false
  - vlm.hybrid_lambda_ctc: 0.6  ->  vlm.hybrid_lambda_ctc: 0.0
  - dir_work: <original>  ->  <original>_noctc

The chapter already has MLP + Pooling-MLP at both with/without CTC; this
adds the matching with/without-CTC pair for Q-Former and KV Multi-View so
all four matched-budget connectors carry the same comparison.

Output:
  configs/_noctc_ch6/<group>/<basename>_noctc.yaml   (4 files)
  scripts/_noctc_ch6_manifest.txt                    (one path per line)
"""
from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent  # work/REWI_work
CFG_ROOT = ROOT / "configs"
OUT_CFG_ROOT = CFG_ROOT / "_noctc_ch6"
OUT_MANIFEST = ROOT / "scripts" / "_noctc_ch6_manifest.txt"

# In manifest order: L1 onhw, L1 word, L2 onhw, L2 word.
SOURCES = [
    "L1_mini_qformer/train-L1-mini-qformer-onhw.yaml",
    "L1_mini_qformer/train-L1-mini-qformer-word.yaml",
    "L2_kv_slim/train-L2-kv-slim-onhw.yaml",
    "L2_kv_slim/train-L2-kv-slim-word.yaml",
]

CTC_TRUE_RE = re.compile(r"^(\s*hybrid_ctc:\s*)true(\s*)$", flags=re.MULTILINE)
LAM_RE = re.compile(r"^(\s*hybrid_lambda_ctc:\s*)0\.6(\s*)$", flags=re.MULTILINE)
DIRWORK_RE = re.compile(r"^(dir_work:\s*)(\S+)$", flags=re.MULTILINE)


def patch(src: str) -> str:
    out, n1 = CTC_TRUE_RE.subn(r"\g<1>false\g<2>", src, count=1)
    if n1 != 1:
        raise RuntimeError("expected exactly one `hybrid_ctc: true` occurrence")
    out, n2 = LAM_RE.subn(r"\g<1>0.0\g<2>", out, count=1)
    if n2 != 1:
        raise RuntimeError("expected exactly one `hybrid_lambda_ctc: 0.6` occurrence")
    out, n3 = DIRWORK_RE.subn(
        lambda mo: f"{mo.group(1)}{mo.group(2)}_noctc", out, count=1
    )
    if n3 != 1:
        raise RuntimeError("expected exactly one `dir_work:` occurrence")
    return out


def main() -> None:
    OUT_CFG_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_lines = []
    for rel in SOURCES:
        src_path = CFG_ROOT / rel
        if not src_path.exists():
            raise FileNotFoundError(src_path)
        src_text = src_path.read_text()
        dst_text = patch(src_text)
        group = Path(rel).parent.name
        base = Path(rel).stem + "_noctc.yaml"
        dst_path = OUT_CFG_ROOT / group / base
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_text(dst_text)
        manifest_lines.append(str(dst_path.relative_to(ROOT)))
        print(f"cloned: {rel} -> {dst_path.relative_to(ROOT)}")

    OUT_MANIFEST.write_text("\n".join(manifest_lines) + "\n")
    print(f"\nwrote manifest ({len(manifest_lines)} configs): {OUT_MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
