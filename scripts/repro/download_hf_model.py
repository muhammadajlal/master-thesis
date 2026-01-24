#!/usr/bin/env python3
"""Download HuggingFace model files into a local directory.

This is intended to support offline-first training on clusters:
- Download once (when internet is available)
- Point `lm_name` at the downloaded directory
- Keep `lm_local_files_only: true` for offline runs

Example:
  python scripts/repro/download_hf_model.py --model t5-small \
    --out-dir assets/hf_models/t5-small

Model page:
  https://huggingface.co/t5-small
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="t5-small",
        help="HuggingFace repo id or local path (default: t5-small)",
    )
    parser.add_argument(
        "--out-dir",
        default="assets/hf_models/t5-small",
        help="Local directory to place the snapshot (default: assets/hf_models/t5-small)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional HF revision (branch/tag/commit).",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        help="Optional HF token (or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN).",
    )

    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print(
            "ERROR: huggingface_hub is not installed. Install with: pip install huggingface_hub\n"
            f"Details: {e}",
            file=sys.stderr,
        )
        return 2

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Downloading {args.model} -> {out_dir}")

    snapshot_download(
        repo_id=args.model,
        local_dir=out_dir,
        local_dir_use_symlinks=False,
        revision=args.revision,
        token=args.token,
    )

    # Small marker for humans
    marker = os.path.join(out_dir, "_DOWNLOADED_FROM_HF.txt")
    with open(marker, "w", encoding="utf-8") as f:
        f.write(f"repo_id={args.model}\n")
        if args.revision:
            f.write(f"revision={args.revision}\n")
        f.write("url=https://huggingface.co/" + args.model + "\n")

    print("Done.")
    print("Tip: set in YAML:\n  lm_name: \"" + out_dir + "\"\n  lm_local_files_only: true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
