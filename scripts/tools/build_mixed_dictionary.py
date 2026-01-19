#!/usr/bin/env python3
"""Build a mixed EN+DE dictionary (words OR sentences) and filter out dataset labels.

This script supports two input types:

- **word**: EN/DE inputs are TSV (rank \t token \t count). The script extracts column 2.
- **sent**: EN/DE inputs are sentence-per-line (optionally prefixed with a numeric rank).

In both cases we:
- normalize (NFKC + lowercase by default)
- optionally filter for "word-like" tokens (word mode)
- remove any item that appears as a ground-truth label in provided dataset JSONs
- deduplicate and shuffle deterministically

Examples
--------

Word dictionary (filter ONHW word labels):

    python3 scripts/tools/build_mixed_dictionary.py \
        --kind word \
        --en assets/dictionaries/word/eng_news_2024_1M-words.txt \
        --de assets/dictionaries/word/deu_news_2024_1M-words.txt \
        --dataset ../../data/onhw_wi_word_rh/train.json --dataset ../../data/onhw_wi_word_rh/val.json \
        --dataset ../../data/onhw_wd_word_rh/train.json --dataset ../../data/onhw_wd_word_rh/val.json \
        --out assets/dictionaries/word/mixed_en_de_no_onhw.txt \
        --out-removed assets/dictionaries/word/removed_due_to_leakage.txt

Sentence dictionary (filter WI sentence labels to avoid leakage):

    python3 scripts/tools/build_mixed_dictionary.py \
        --kind sent \
        --en assets/dictionaries/sent/eng_news_2024_1M-sentences.txt \
        --de assets/dictionaries/sent/deu_news_2024_1M-sentences.txt \
        --dataset ../../data/wi_sent_hw6_meta \
        --out assets/dictionaries/sent/mixed_en_de_no_wi_sent_hw6_meta.txt \
        --out-removed assets/dictionaries/sent/removed_due_to_leakage.txt
"""

from __future__ import annotations

import argparse
import json
import random
import unicodedata
from pathlib import Path


def _normalize(token: str, *, lowercase: bool, nfkc: bool, normalize_ws: bool) -> str:
    token = token.strip()
    if nfkc:
        token = unicodedata.normalize("NFKC", token)
    if lowercase:
        token = token.lower()
    if normalize_ws:
        # Collapse whitespace runs only when needed.
        # Avoid calling split/join for the common case (single spaces already).
        if (
            "\t" in token
            or "\r" in token
            or "\n" in token
            or "  " in token
            or "\u00a0" in token
        ):
            token = " ".join(token.split())
    return token


def _has_letter(s: str) -> bool:
    return any(unicodedata.category(ch).startswith("L") for ch in s)


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

    if not _has_letter(token):
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


def _read_tsv_wordlist(
    path: Path,
    *,
    lowercase: bool,
    nfkc: bool,
    normalize_ws: bool,
    min_count: int,
) -> list[str]:
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

            token = _normalize(token, lowercase=lowercase, nfkc=nfkc, normalize_ws=normalize_ws)
            if _is_word_like(token):
                words.append(token)
    return words


def _strip_leading_rank(line: str) -> str:
    """Drop an optional leading numeric rank.

    Handles lines like:
      "123  some sentence"
      "123\tsome token"
    """

    s = line.lstrip()
    if not s:
        return ""
    parts = s.split(None, 1)
    if len(parts) == 2 and parts[0].isdigit():
        return parts[1]
    return s


def _read_sentence_list(
    path: Path,
    *,
    lowercase: bool,
    nfkc: bool,
    normalize_ws: bool,
    require_letter: bool,
) -> list[str]:
    sents: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue

            text = _strip_leading_rank(line)
            text = text.strip()
            # The provided news sentence lists seem to prefix many lines with '$'
            # (e.g. "$An den Finanzmärkten ..."). Strip it.
            if text.startswith("$"):
                text = text[1:].lstrip()

            text = _normalize(text, lowercase=lowercase, nfkc=nfkc, normalize_ws=normalize_ws)
            if not text:
                continue
            if require_letter and not _has_letter(text):
                continue

            sents.append(text)
    return sents


def _iter_sentence_items(
    path: Path,
    *,
    lowercase: bool,
    nfkc: bool,
    normalize_ws: bool,
    require_letter: bool,
):
    """Yield normalized sentences one-by-one (streaming)."""

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue

            text = _strip_leading_rank(line)
            text = text.strip()
            if text.startswith("$"):
                text = text[1:].lstrip()

            text = _normalize(text, lowercase=lowercase, nfkc=nfkc, normalize_ws=normalize_ws)
            if not text:
                continue
            if require_letter and not _has_letter(text):
                continue

            yield text


def _iter_dataset_jsons(path: Path) -> list[Path]:
    """Allow passing either a dataset JSON file or a dataset directory.

    If a directory is passed, we auto-include train.json/val.json when present.
    """

    path = Path(path)
    if path.is_dir():
        out: list[Path] = []
        for name in ["train.json", "val.json"]:
            p = path / name
            if p.exists():
                out.append(p)
        return out
    return [path]


def _extract_labels_from_dataset_json(
    path: Path,
    *,
    lowercase: bool,
    nfkc: bool,
    normalize_ws: bool,
) -> set[str]:
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
            token = _normalize(raw, lowercase=lowercase, nfkc=nfkc, normalize_ws=normalize_ws)
            if token:
                forbidden.add(token)

    return forbidden


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--kind",
        choices=["word", "sent"],
        default="word",
        help="Input/output type: word (TSV token list) or sent (sentence-per-line list)",
    )
    ap.add_argument("--en", type=Path, required=True, help="English input file (TSV wordlist for kind=word, text sentences for kind=sent)")
    ap.add_argument("--de", type=Path, required=True, help="German input file (TSV wordlist for kind=word, text sentences for kind=sent)")
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
    ap.add_argument(
        "--keep-whitespace",
        action="store_true",
        help="Do not normalize whitespace (by default, whitespace runs are collapsed)",
    )
    ap.add_argument("--min-count", type=int, default=1, help="Minimum count column (if present)")
    args = ap.parse_args()

    lowercase = not args.no_lower
    nfkc = not args.no_nfkc
    normalize_ws = not args.keep_whitespace

    # 1) forbidden set from datasets
    forbidden: set[str] = set()
    for ds in args.dataset:
        for json_path in _iter_dataset_jsons(ds):
            forbidden |= _extract_labels_from_dataset_json(
                json_path,
                lowercase=lowercase,
                nfkc=nfkc,
                normalize_ws=normalize_ws,
            )

    # 2) read sources + build combined set
    combined: set[str] = set()
    en_raw = 0
    de_raw = 0

    if args.kind == "word":
        en_items = _read_tsv_wordlist(
            args.en,
            lowercase=lowercase,
            nfkc=nfkc,
            normalize_ws=normalize_ws,
            min_count=args.min_count,
        )
        de_items = _read_tsv_wordlist(
            args.de,
            lowercase=lowercase,
            nfkc=nfkc,
            normalize_ws=normalize_ws,
            min_count=args.min_count,
        )
        en_raw = len(en_items)
        de_raw = len(de_items)
        combined.update(en_items)
        combined.update(de_items)
    else:
        # Sentence mode: stream (files are large)
        for _ in _iter_sentence_items(
            args.en,
            lowercase=lowercase,
            nfkc=nfkc,
            normalize_ws=normalize_ws,
            require_letter=True,
        ):
            en_raw += 1
            combined.add(_)
        for _ in _iter_sentence_items(
            args.de,
            lowercase=lowercase,
            nfkc=nfkc,
            normalize_ws=normalize_ws,
            require_letter=True,
        ):
            de_raw += 1
            combined.add(_)

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
    print(f"  kind={args.kind}")
    print(f"  en_raw={en_raw} de_raw={de_raw}")
    print(f"  forbidden_from_datasets={len(forbidden)}")
    print(f"  unique_before_filter={before}")
    print(f"  unique_after_filter={after}")
    if args.out_removed is not None:
        print(f"  removed_due_to_leakage={len(removed_due_to_leakage)} -> {args.out_removed}")
    print(f"  wrote={args.out} (lines={len(words)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
