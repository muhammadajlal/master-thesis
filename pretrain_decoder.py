import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from rewi.manager import RunManager
from rewi.model import build_decoder
from rewi.tokenizer import BPETokenizer, CharTokenizer


def _seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _levenshtein(a: str, b: str) -> int:
    # DP edit distance (OK for words)
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


@dataclass
class Batch:
    y_inp: torch.Tensor  # (B, T)
    y_tgt: torch.Tensor  # (B, T)


class WordListDataset(Dataset):
    def __init__(
        self,
        path_words: str | Path,
        *,
        max_words: int | None,
        seed: int,
        allowed_chars: set[str] | None = None,
        unknown_policy: str = "drop",  # drop|strip
    ) -> None:
        path_words = Path(path_words)
        if unknown_policy not in {"drop", "strip"}:
            raise ValueError(f"unknown_policy must be 'drop' or 'strip', got: {unknown_policy}")

        words: list[str] = []
        with path_words.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                w = line.strip("\n").strip()
                if not w:
                    continue

                if allowed_chars is not None:
                    if unknown_policy == "drop":
                        if any(ch not in allowed_chars for ch in w):
                            continue
                    else:  # strip
                        w2 = "".join(ch for ch in w if ch in allowed_chars)
                        if not w2:
                            continue
                        w = w2

                words.append(w)
                if max_words is not None and len(words) >= max_words:
                    break

        rng = random.Random(seed)
        rng.shuffle(words)
        self.words = words

    def __len__(self) -> int:
        return len(self.words)

    def __getitem__(self, idx: int) -> str:
        return self.words[idx]


def _make_split(n: int, val_ratio: float, *, seed: int) -> tuple[list[int], list[int]]:
    idxs = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    n_val = max(1, int(n * val_ratio))
    return idxs[n_val:], idxs[:n_val]


def _collate_words(
    words: list[str],
    *,
    tokenizer,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    max_len: int,
) -> Batch:
    # Build y = [BOS] + tokens + [EOS], then teacher-force
    seqs: list[list[int]] = []
    for w in words:
        ids = tokenizer.encode(w)
        ids = [bos_id] + ids[: max_len - 2] + [eos_id]
        seqs.append(ids)

    # pad to max T
    T = max(len(s) for s in seqs)
    y = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        y[i, : len(s)] = torch.tensor(s, dtype=torch.long)

    y_inp = y[:, :-1]
    y_tgt = y[:, 1:]
    return Batch(y_inp=y_inp, y_tgt=y_tgt)


class CategoriesTokenizer:
    """Character-ID mapping compatible with the existing CNN-AR 'no tokenizer' pipeline.

    - IDs 0..(base-1) are exactly indices in cfgs.categories
    - PAD/BOS/EOS are appended at the end: base, base+1, base+2
    """

    def __init__(self, categories: list[str]):
        self.categories = categories
        self.base = len(categories)
        self.PAD = self.base
        self.BOS = self.base + 1
        self.EOS = self.base + 2
        self.vocab_size = self.base + 3

        self.stoi = {c: i for i, c in enumerate(categories) if c != ""}
        self.itos = {i: c for i, c in enumerate(categories) if c != ""}

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for ch in text:
            if ch not in self.stoi:
                raise KeyError(ch)
            ids.append(self.stoi[ch])
        return ids

    def decode(self, ids: list[int]) -> str:
        out: list[str] = []
        for i in ids:
            if i in (self.PAD, self.BOS, self.EOS):
                continue
            if i == 0:
                continue
            out.append(self.itos.get(i, ""))
        return "".join(out)


@torch.no_grad()
def _greedy_decode(decoder, tokenizer, *, device: torch.device, bos_id: int, eos_id: int, max_len: int) -> str:
    # Decode one word with dummy memory
    y = torch.tensor([[bos_id]], dtype=torch.long, device=device)
    memory = torch.zeros((1, 1, decoder.d_model), device=device)
    mem_pad = torch.zeros((1, 1), dtype=torch.bool, device=device)

    out_ids: list[int] = []
    for _ in range(max_len):
        logits = decoder(y, memory, mem_pad_mask=mem_pad)  # (1, T, V)
        next_id = int(torch.argmax(logits[0, -1], dim=-1).item())
        if next_id == eos_id:
            break
        out_ids.append(next_id)
        y = torch.cat([y, torch.tensor([[next_id]], device=device)], dim=1)

    return tokenizer.decode(out_ids)


@torch.no_grad()
def evaluate_greedy_cer_wer(
    decoder,
    tokenizer,
    loader,
    *,
    device: torch.device,
    max_decode_len: int,
    max_samples: int | None,
) -> dict:
    """Greedy decode CER/WER against random target words.

    IMPORTANT: In decoder-only pretraining we do not condition on input features
    (memory is constant), so the decoder cannot know *which* word to output.
    Therefore CER/WER here is mostly a diagnostic and often stays poor.
    Prefer `val_loss` / `val_ppl` for tracking learning progress.
    """
    decoder.eval()

    total_edits = 0
    total_chars = 0
    total_words = 0
    word_errors = 0

    seen = 0
    for batch_words in loader:
        for w in batch_words:
            pred = _greedy_decode(
                decoder,
                tokenizer,
                device=device,
                bos_id=tokenizer.BOS,
                eos_id=tokenizer.EOS,
                max_len=max_decode_len,
            )
            tgt = w
            total_edits += _levenshtein(pred, tgt)
            total_chars += max(1, len(tgt))
            total_words += 1
            if pred != tgt:
                word_errors += 1

            seen += 1
            if max_samples is not None and seen >= max_samples:
                break
        if max_samples is not None and seen >= max_samples:
            break

    cer = total_edits / max(1, total_chars)
    wer = word_errors / max(1, total_words)
    return {
        "character_error_rate": float(cer),
        "word_error_rate": float(wer),
    }


@torch.no_grad()
def evaluate_teacher_forced_loss(decoder, tokenizer, loader, *, device: torch.device) -> dict:
    """Compute next-token cross-entropy on the validation split (teacher forcing)."""
    decoder.eval()

    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        y_inp = batch.y_inp.to(device, non_blocking=True)
        y_tgt = batch.y_tgt.to(device, non_blocking=True)

        B = y_inp.size(0)
        memory = torch.zeros((B, 1, decoder.d_model), device=device)
        mem_pad = torch.zeros((B, 1), dtype=torch.bool, device=device)

        logits = decoder(y_inp, memory, mem_pad_mask=mem_pad)  # (B, T, V)
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_tgt = y_tgt.reshape(-1)

        # Count non-pad tokens
        valid = (flat_tgt != int(tokenizer.PAD))
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            continue

        loss = F.cross_entropy(flat_logits, flat_tgt, ignore_index=int(tokenizer.PAD), reduction="sum")
        total_loss += float(loss.item())
        total_tokens += n_valid

    if total_tokens == 0:
        # Extremely unlikely unless dataset is empty.
        return {"val_loss": float("inf"), "val_ppl": float("inf")}

    val_loss = total_loss / total_tokens
    val_ppl = math.exp(min(20.0, val_loss))  # clamp to avoid overflow spam
    return {"val_loss": float(val_loss), "val_ppl": float(val_ppl)}


def main(cfgs: argparse.Namespace) -> None:
    _seed_all(int(getattr(cfgs, "seed", 1337)))

    requested_device = str(getattr(cfgs, "device", "cpu"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("[pretrain_decoder] CUDA requested but not available; falling back to CPU")
        requested_device = "cpu"
    device = torch.device(requested_device)

    # Tokenizer selection
    tok_cfg = getattr(cfgs, "tokenizer", {}) or {}
    tok_type = str(tok_cfg.get("type", "categories"))

    if tok_type == "sentencepiece":
        tok_model = tok_cfg["model"]
        tokenizer = BPETokenizer(tok_model)
        vocab_size = tokenizer.vocab_size
    elif tok_type == "char":
        vocab_path = tok_cfg.get("vocab")
        if vocab_path:
            tokenizer = CharTokenizer.load(vocab_path)
        else:
            # Support both wordlist_path and sentlist_path
            textlist_path = getattr(cfgs, "wordlist_path", None) or getattr(cfgs, "sentlist_path", None)
            if textlist_path is None:
                raise ValueError("tokenizer.type=char requires either cfgs.wordlist_path or cfgs.sentlist_path")
            tokenizer = CharTokenizer.build_from_text_file(
                textlist_path,
                lowercase=bool(tok_cfg.get("lowercase", True)),
                nfkc=bool(tok_cfg.get("nfkc", True)),
                max_lines=tok_cfg.get("max_vocab_lines"),
            )

        # Save vocab next to the work dir for downstream reuse
        out_vocab = Path(cfgs.dir_work) / str(cfgs.idx_fold) / "tokenizer_char_vocab.json"
        tokenizer.save(out_vocab)
        vocab_size = tokenizer.vocab_size
    elif tok_type == "categories":
        # Use the same character IDs as CNN-AR "no tokenizer" mode.
        # Prefer loading categories from an existing training YAML to avoid duplication.
        categories = tok_cfg.get("categories")
        if categories is None:
            src_yaml = tok_cfg.get("categories_from_yaml")
            if not src_yaml:
                raise ValueError("tokenizer.type=categories requires tokenizer.categories or tokenizer.categories_from_yaml")
            with open(src_yaml, "r") as f:
                y = yaml.safe_load(f)
            key = tok_cfg.get("categories_key", "categories")
            categories = y.get(key)
        if not isinstance(categories, list) or not categories:
            raise ValueError("Invalid categories for categories tokenizer")

        tokenizer = CategoriesTokenizer(categories)
        vocab_size = tokenizer.vocab_size
    else:
        raise ValueError(f"Unknown tokenizer.type: {tok_type}")

    # Build decoder
    decoder = build_decoder(
        dim_in=0,
        num_cls=vocab_size,
        arch=cfgs.arch_de,
        use_gated_attention=bool(getattr(cfgs, "use_gated_attention", False)),
        gating_type=str(getattr(cfgs, "gating_type", "elementwise")),
    ).to(device)

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=float(cfgs.lr), weight_decay=float(getattr(cfgs, "weight_decay", 0.0)))

    # Optional cosine schedule
    lr_scheduler = None
    if getattr(cfgs, "use_cosine", False):
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfgs.epoch))

    # Dataset + split
    allowed_chars = None
    unknown_policy = str(getattr(cfgs, "unknown_policy", "drop"))
    if isinstance(tokenizer, CategoriesTokenizer):
        allowed_chars = set(tokenizer.stoi.keys())

    # Support both wordlist_path and sentlist_path (sentence pretraining)
    textlist_path = getattr(cfgs, "wordlist_path", None) or getattr(cfgs, "sentlist_path", None)
    if textlist_path is None:
        raise ValueError("Config must specify either 'wordlist_path' or 'sentlist_path'")

    ds = WordListDataset(
        textlist_path,
        max_words=getattr(cfgs, "max_words", None),
        seed=int(getattr(cfgs, "seed", 1337)),
        allowed_chars=allowed_chars,
        unknown_policy=unknown_policy,
    )
    idx_train, idx_val = _make_split(len(ds), float(cfgs.val_ratio), seed=int(getattr(cfgs, "seed", 1337)))
    ds_train = torch.utils.data.Subset(ds, idx_train)
    ds_val = torch.utils.data.Subset(ds, idx_val)

    collate = lambda ws: _collate_words(
        ws,
        tokenizer=tokenizer,
        pad_id=tokenizer.PAD,
        bos_id=tokenizer.BOS,
        eos_id=tokenizer.EOS,
        max_len=int(cfgs.max_len),
    )

    train_loader = DataLoader(
        ds_train,
        batch_size=int(cfgs.batch_size),
        shuffle=True,
        num_workers=int(getattr(cfgs, "num_workers", 4)),
        pin_memory=True,
        collate_fn=collate,
        drop_last=True,
    )

    # Validation loaders:
    # - teacher-forced loss uses the same collate as training
    # - greedy CER/WER uses raw word strings
    val_loader_loss = DataLoader(
        ds_val,
        batch_size=int(getattr(cfgs, "batch_size", 512)),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate,
        drop_last=False,
    )
    val_loader_words = DataLoader(
        ds_val,
        batch_size=int(getattr(cfgs, "eval_batch_words", 512)),
        shuffle=False,
        num_workers=0,
        collate_fn=lambda ws: ws,
    )

    manager = RunManager(cfgs)

    def _load_best_val_loss_from_existing_results(dir_work: str, idx_fold: int) -> float | None:
        fold_dir = Path(dir_work) / str(idx_fold)
        if not fold_dir.exists():
            return None

        candidates = sorted(
            fold_dir.glob("train_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            return None

        try:
            with candidates[0].open("r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            return None

        best = None
        for k, v in results.items():
            if not isinstance(k, str) or not k.isdigit():
                continue
            if not isinstance(v, dict):
                continue
            ev = v.get("evaluation")
            if not isinstance(ev, dict):
                continue
            val = ev.get("val_loss")
            if val is None:
                continue
            try:
                val_f = float(val)
            except Exception:
                continue
            best = val_f if best is None else min(best, val_f)
        return best

    # Resume support (optional)
    start_epoch = 1
    resume = bool(getattr(cfgs, "resume", False))
    resume_checkpoint = getattr(cfgs, "resume_checkpoint", None)
    if resume and not resume_checkpoint:
        resume_checkpoint = str(Path(cfgs.dir_work) / str(cfgs.idx_fold) / "checkpoints" / "last.pth")

    if resume_checkpoint:
        ckpt_path = Path(str(resume_checkpoint))
        if not ckpt_path.exists():
            raise FileNotFoundError(f"resume_checkpoint not found: {ckpt_path}")

        ckpt = torch.load(str(ckpt_path), map_location=device)
        if isinstance(ckpt, dict):
            state_model = ckpt.get("model")
            state_optimizer = ckpt.get("optimizer")
            state_lr_scheduler = ckpt.get("lr_scheduler")
            epoch_done = int(ckpt.get("epoch", 0))

            if state_model:
                decoder.load_state_dict(state_model)
            if state_optimizer:
                optimizer.load_state_dict(state_optimizer)
            if lr_scheduler and state_lr_scheduler:
                lr_scheduler.load_state_dict(state_lr_scheduler)

            start_epoch = max(1, epoch_done + 1)
            print(f"[pretrain_decoder] Resuming from {ckpt_path} (epoch {epoch_done} -> start {start_epoch})")
        else:
            raise ValueError(f"Invalid checkpoint format at {ckpt_path}")

    # Initialize best_* so we don't overwrite best_loss.pth immediately on resume
    best_loss = math.inf
    if start_epoch > 1:
        prev_best = _load_best_val_loss_from_existing_results(str(cfgs.dir_work), int(cfgs.idx_fold))
        if prev_best is not None:
            best_loss = float(prev_best)

    best_cer = math.inf
    best_wer = math.inf

    for epoch in range(start_epoch, int(cfgs.epoch) + 1):
        decoder.train()
        manager.initialize_epoch(epoch, num_iter=len(train_loader), val=False)

        max_train_iters = getattr(cfgs, "max_train_iters", None)
        if max_train_iters is not None:
            max_train_iters = int(max_train_iters)

        for it, batch in enumerate(train_loader):
            if max_train_iters is not None and it >= max_train_iters:
                break
            y_inp = batch.y_inp.to(device, non_blocking=True)
            y_tgt = batch.y_tgt.to(device, non_blocking=True)

            # Dummy memory (constant) so ARDecoder forward works
            B = y_inp.size(0)
            memory = torch.zeros((B, 1, decoder.d_model), device=device)
            mem_pad = torch.zeros((B, 1), dtype=torch.bool, device=device)

            logits = decoder(y_inp, memory, mem_pad_mask=mem_pad)  # (B, T, V)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y_tgt.reshape(-1),
                ignore_index=int(tokenizer.PAD),
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), float(getattr(cfgs, "clip_grad", 1.0)))
            optimizer.step()

            lr = optimizer.param_groups[0]["lr"]
            manager.update_iteration(it, float(loss.item()), lr=float(lr))

        manager.summarize_epoch()

        # Always save rolling resume checkpoint
        manager.save_checkpoint(
            decoder.state_dict(),
            optimizer.state_dict(),
            lr_scheduler.state_dict() if lr_scheduler else None,
            filename="last.pth",
        )

        # Evaluate
        manager.initialize_epoch(epoch, num_iter=1, val=True)
        metrics = {}

        # Primary metric for decoder-only pretraining: teacher-forced validation loss
        metrics.update(evaluate_teacher_forced_loss(decoder, tokenizer, val_loader_loss, device=device))

        # Optional diagnostic: greedy CER/WER (often stays poor for decoder-only)
        if bool(getattr(cfgs, "eval_greedy", True)):
            metrics.update(
                evaluate_greedy_cer_wer(
                    decoder,
                    tokenizer,
                    val_loader_words,
                    device=device,
                    max_decode_len=int(getattr(cfgs, "max_decode_len", cfgs.max_len)),
                    max_samples=getattr(cfgs, "eval_max_samples", 2000),
                )
            )
        manager.update_evaluation(metrics)

        val_loss = float(metrics.get("val_loss", math.inf))
        if val_loss < best_loss:
            best_loss = val_loss
            manager.save_checkpoint(
                decoder.state_dict(),
                optimizer.state_dict(),
                lr_scheduler.state_dict() if lr_scheduler else None,
                filename="best_loss.pth",
            )

        cer = metrics.get("character_error_rate", None)
        wer = metrics.get("word_error_rate", None)
        if cer is not None and float(cer) < best_cer:
            best_cer = float(cer)
            manager.save_checkpoint(
                decoder.state_dict(),
                optimizer.state_dict(),
                lr_scheduler.state_dict() if lr_scheduler else None,
                filename="best_cer.pth",
            )

        if wer is not None and float(wer) < best_wer:
            best_wer = float(wer)
            manager.save_checkpoint(
                decoder.state_dict(),
                optimizer.state_dict(),
                lr_scheduler.state_dict() if lr_scheduler else None,
                filename="best_wer.pth",
            )

        if lr_scheduler:
            lr_scheduler.step()

    manager.summarize_evaluation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain AR decoder on text wordlist.")
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfgs = yaml.safe_load(f)
        cfgs = argparse.Namespace(**cfgs)

    main(cfgs)
