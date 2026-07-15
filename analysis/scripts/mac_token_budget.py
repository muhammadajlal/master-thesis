#!/usr/bin/env python3
"""Aggregate the OnHW target lengths used to set fixed MAC profiling budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(lengths: list[int]) -> dict[str, object]:
    return {
        "count": len(lengths),
        "mean": statistics.mean(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "distribution": {str(key): value for key, value in sorted(Counter(lengths).items())},
        "rounded_up_profile_budget": math.ceil(statistics.mean(lengths)),
    }


def analyze(manifest: Path, tokenizer_dir: Path) -> dict[str, object]:
    data = json.loads(manifest.read_text())
    annotations = data["annotations"]
    records = [
        item
        for fold in sorted(annotations, key=int)
        for item in annotations[fold]
    ]
    labels = [str(item["label"]) for item in records]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    gpt2_lengths = [
        len(tokenizer.encode(label, add_special_tokens=False))
        for label in labels
    ]
    character_lengths = [len(label) for label in labels]

    return {
        "scope": "All five OnHW-words500 writer-independent validation folds.",
        "privacy": "Only aggregate counts are emitted; labels are not stored.",
        "length_definition": "Target text only; BOS, EOS, padding, prompts, and sensor tokens are excluded.",
        "source": {
            "manifest": "data/onhw_wi_word_rh/val.json",
            "manifest_sha256": sha256(manifest),
            "tokenizer": "assets/hf_models/gpt2",
            "tokenizer_json_sha256": sha256(tokenizer_dir / "tokenizer.json"),
        },
        "character_target_length": summarize(character_lengths),
        "gpt2_target_token_length": summarize(gpt2_lengths),
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    project_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "data" / "onhw_wi_word_rh" / "val.json",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=repo_root / "assets" / "hf_models" / "gpt2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "results" / "thesis_mac_token_budget.json",
    )
    args = parser.parse_args()

    report = analyze(args.manifest.resolve(), args.tokenizer.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
