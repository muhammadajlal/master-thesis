#!/usr/bin/env python3
"""Build a mixed EN+DE word list and filter out any words that appear in ONHW datasets.

Your EN/DE files are TSV (rank \t token \t count). This script:
- Extracts the token column
- Normalizes tokens (NFKC + lowercase by default)
- Keeps only "word-like" tokens (letters with optional internal '-' or ''')
- Removes any token that appears as a ground-truth label in the provided dataset JSONs
- Deduplicates and shuffles deterministically

Example:
  python3 tools/build_mixed_dictionary.py \
    --en assets/dictionaries/eng_news_2024_1M-words.txt \
    --de assets/dictionaries/deu_news_2024_1M-words.txt \
    --dataset ../../data/onhw_wi_word_rh/train.json --dataset ../../data/onhw_wi_word_rh/val.json \
    --dataset ../../data/onhw_wd_word_rh/train.json --dataset ../../data/onhw_wd_word_rh/val.json \
        --out assets/dictionaries/mixed_en_de_no_onhw.txt \
        --out-removed assets/dictionaries/removed_due_to_leakage.txt
"""

from __future__ import annotations

import argparse
import json
import random
import unicodedata
from pathlib import Path


def _normalize(token: str, *, lowercase: bool, nfkc: bool) -> str:
    token = token.strip()
    if nfkc:
        token = unicodedata.normalize("NFKC", token)
    if lowercase:
        token = token.lower()
    return token


def _is_word_like(token: str) -> bool:
    """Heuristic filter for dictionary tokens.

    Keep strings that:
    - contain at least one Unicode letter
    - consist only of letters/combining-marks plus optional internal '-' or '''

    This drops punctuation-only tokens like "," or "€" which appear in the news vocab.
    """

    if not token:
        return False

    def is_letter_or_mark(ch: str) -> bool:
        cat = unicodedata.category(ch)
        return cat.startswith("L") or cat.startswith("M")

    has_letter = any(unicodedata.category(ch).startswith("L") for ch in token)
    if not has_letter:
        return False

    for i, ch in enumerate(token):
        if is_letter_or_mark(ch):
            continue
        if ch in {"-", "'"}:
            # allow only internal hyphen/apostrophe
            if i == 0 or i == len(token) - 1:
                return False
            continue
        return False

    return True


def _read_tsv_wordlist(path: Path, *, lowercase: bool, nfkc: bool, min_count: int) -> list[str]:
    words: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            token = parts[1]
            count = None
            if len(parts) >= 3:
                try:
                    count = int(parts[2])
                except ValueError:
                    count = None
            if count is not None and count < min_count:
                continue

            token = _normalize(token, lowercase=lowercase, nfkc=nfkc)
            if _is_word_like(token):
                words.append(token)
    return words


def _extract_labels_from_dataset_json(path: Path, *, lowercase: bool, nfkc: bool) -> set[str]:
    """Extract the ground-truth word labels from a MSCOCO-like dataset JSON.

    For ONHW word datasets, the label field is `label` and `annotations` is a dict keyed by fold.
    """

    obj = json.loads(path.read_text(encoding="utf-8"))
    ann = obj.get("annotations", {})

    forbidden: set[str] = set()

    if isinstance(ann, dict):
        fold_values = ann.values()
    elif isinstance(ann, list):
        fold_values = [ann]
    else:
        fold_values = []

    for items in fold_values:
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            # Common possibilities across datasets
            raw = it.get("label") or it.get("text") or it.get("word")
            if not isinstance(raw, str):
                continue
            token = _normalize(raw, lowercase=lowercase, nfkc=nfkc)
            if token:
                forbidden.add(token)

    return forbidden


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--en", type=Path, required=True, help="English TSV wordlist")
    ap.add_argument("--de", type=Path, required=True, help="German TSV wordlist")
    ap.add_argument("--dataset", type=Path, action="append", default=[], help="Dataset JSON(s) to exclude labels from")
    ap.add_argument("--out", type=Path, required=True, help="Output .txt (one word per line)")
    ap.add_argument(
        "--out-removed",
        type=Path,
        default=None,
        help="Optional output .txt listing tokens removed due to leakage (one token per line)",
    )
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-lower", action="store_true", help="Do not lowercase")
    ap.add_argument("--no-nfkc", action="store_true", help="Do not apply NFKC normalization")
    ap.add_argument("--min-count", type=int, default=1, help="Minimum count column (if present)")
    args = ap.parse_args()

    lowercase = not args.no_lower
    nfkc = not args.no_nfkc

    # 1) forbidden set from datasets
    forbidden: set[str] = set()
    for ds in args.dataset:
        forbidden |= _extract_labels_from_dataset_json(ds, lowercase=lowercase, nfkc=nfkc)

    # 2) read wordlists
    en_words = _read_tsv_wordlist(args.en, lowercase=lowercase, nfkc=nfkc, min_count=args.min_count)
    de_words = _read_tsv_wordlist(args.de, lowercase=lowercase, nfkc=nfkc, min_count=args.min_count)

    # 3) dedupe + filter leakage
    combined = set(en_words)
    combined |= set(de_words)

    before_set = combined
    combined = {w for w in before_set if w not in forbidden}
    removed_due_to_leakage = before_set - combined
    before = len(before_set)
    after = len(combined)

    # 4) shuffle deterministically and write
    words = list(combined)
    random.Random(args.seed).shuffle(words)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(words) + "\n", encoding="utf-8")

    if args.out_removed is not None:
        args.out_removed.parent.mkdir(parents=True, exist_ok=True)
        # Deterministic ordering for auditing
        removed_sorted = sorted(removed_due_to_leakage)
        args.out_removed.write_text("\n".join(removed_sorted) + ("\n" if removed_sorted else ""), encoding="utf-8")

    print("[build_mixed_dictionary]")
    print(f"  en_raw={len(en_words)} de_raw={len(de_words)}")
    print(f"  forbidden_from_datasets={len(forbidden)}")
    print(f"  unique_before_filter={before}")
    print(f"  unique_after_filter={after}")
    if args.out_removed is not None:
        print(f"  removed_due_to_leakage={len(removed_due_to_leakage)} -> {args.out_removed}")
    print(f"  wrote={args.out} (lines={len(words)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
