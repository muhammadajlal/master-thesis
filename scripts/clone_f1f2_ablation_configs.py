#!/usr/bin/env python3
"""Clone F1/F2 frozen-encoder configs into the 2x2x2 matched-simple ablation.

Grid: 2 encoder sources x 2 decoders x 2 datasets = 8 configs.

Encoder sources (both single-supervision, no auxiliary CTC):
  qenc  = L1 mini Q-Former trained WITHOUT auxiliary CTC (noctc runs)
  hwenc = HWRFormer-xs AR-only baseline (Baseline-AR-XS-blconv_b)

Decoders:
  F1 = MLP connector + LoRA GPT-2 (multimodal head), cloned from
       configs/F1_frozen_enc_vlm/train-F1-mlp-{onhw,word}.yaml
  F2 = HWRFormer decoder trained from scratch, cloned from
       configs/F2_pretrained_enc/train-F2-{onhw,word}.yaml
       with arch_de switched ar_transformer_s -> ar_transformer_xs so the
       decoder matches the thesis's primary HWRFormer configuration.

Each clone changes ONLY:
  - checkpoint: points at the matching encoder bank (keeps {fold} template)
  - dir_work:   distinct output path per cell
  - arch_de:    F2 only, s -> xs

Output:
  configs/_f1f2_ablation/<cell>.yaml  (8 files)
  scripts/_f1f2_ablation_manifest.txt (one path per line, 8 lines)
"""
from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
CFG_ROOT = ROOT / "configs"
OUT_CFG_ROOT = CFG_ROOT / "_f1f2_ablation"
OUT_MANIFEST = ROOT / "scripts" / "_f1f2_ablation_manifest.txt"

ENC_BANKS = {
    "qenc": {
        "onhw": "assets/pretrained_encoders/qformer_mini_noctc_onhw",
        "word": "assets/pretrained_encoders/qformer_mini_noctc_word",
    },
    "hwenc": {
        "onhw": "assets/pretrained_encoders/hwrformer_xs_onhw",
        "word": "assets/pretrained_encoders/hwrformer_xs_word",
    },
}

SOURCES = {
    ("F1", "onhw"): "F1_frozen_enc_vlm/train-F1-mlp-onhw.yaml",
    ("F1", "word"): "F1_frozen_enc_vlm/train-F1-mlp-word.yaml",
    ("F2", "onhw"): "F2_pretrained_enc/train-F2-onhw.yaml",
    ("F2", "word"): "F2_pretrained_enc/train-F2-word.yaml",
}

CKPT_RE = re.compile(r"^(checkpoint:\s*)(\S+)$", flags=re.MULTILINE)
DIRWORK_RE = re.compile(r"^(dir_work:\s*)(\S+)$", flags=re.MULTILINE)
ARCHDE_S_RE = re.compile(r"^(arch_de:\s*)'ar_transformer_s'(\s*)$", flags=re.MULTILINE)


def patch(src: str, decoder: str, enc_key: str, dataset: str) -> str:
    bank = ENC_BANKS[enc_key][dataset]
    new_ckpt = f"/home/woody/iwso/iwso214h/imu-hwr/work/REWI_work/{bank}/encoder_fold{{fold}}.pth"

    out, n1 = CKPT_RE.subn(lambda m: f"{m.group(1)}{new_ckpt}", src, count=1)
    if n1 != 1:
        raise RuntimeError("expected exactly one checkpoint line")

    out, n2 = DIRWORK_RE.subn(
        lambda m: f"{m.group(1)}{m.group(2)}_{enc_key}_ablation", out, count=1
    )
    if n2 != 1:
        raise RuntimeError("expected exactly one dir_work line")

    if decoder == "F2":
        out, n3 = ARCHDE_S_RE.subn(r"\g<1>'ar_transformer_xs'\g<2>", out, count=1)
        if n3 != 1:
            raise RuntimeError("expected exactly one arch_de: 'ar_transformer_s' line in F2")

    return out


def main() -> None:
    OUT_CFG_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_lines = []
    # Deterministic manifest order: F1-qenc-onhw, F1-qenc-word, F1-hwenc-onhw,
    # F1-hwenc-word, F2-qenc-onhw, F2-qenc-word, F2-hwenc-onhw, F2-hwenc-word.
    for decoder in ("F1", "F2"):
        for enc_key in ("qenc", "hwenc"):
            for dataset in ("onhw", "word"):
                src_path = CFG_ROOT / SOURCES[(decoder, dataset)]
                if not src_path.exists():
                    raise FileNotFoundError(src_path)
                dst_text = patch(src_path.read_text(), decoder, enc_key, dataset)
                dst_path = OUT_CFG_ROOT / f"train-{decoder}-{enc_key}-{dataset}.yaml"
                dst_path.write_text(dst_text)
                manifest_lines.append(str(dst_path.relative_to(ROOT)))
                print(f"cloned: {SOURCES[(decoder, dataset)]} + {enc_key} -> {dst_path.relative_to(ROOT)}")

    OUT_MANIFEST.write_text("\n".join(manifest_lines) + "\n")
    print(f"\nwrote manifest ({len(manifest_lines)} configs): {OUT_MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
