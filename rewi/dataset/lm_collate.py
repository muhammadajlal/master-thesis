# rewi/dataset/lm_collate.py
from __future__ import annotations
from typing import List, Tuple, Optional

import torch

def decode_ids_to_text(y: torch.Tensor, len_y: torch.Tensor, categories: List[str], pad_id: int, eos_id: Optional[int] = None) -> List[str]:
    # y: (B, Smax)
    texts = []
    B = y.size(0)
    for b in range(B):
        L = int(len_y[b].item())
        ids = y[b, :L].tolist()
        chars = []
        for t in ids:
            if eos_id is not None and t == eos_id:
                break
            if t == pad_id:
                continue
            # skip CTC blank if present as categories[0] == ""
            if 0 <= t < len(categories) and t != 0:
                chars.append(categories[t])
        texts.append("".join(chars))
    return texts

def _chars_to_strings(y, len_y, categories, pad_id):
    """Convert char-level token IDs to Python strings."""
    texts = []
    for i in range(y.size(0)):
        seq = y[i, :len_y[i]].tolist()
        # drop pad ids
        seq = [t for t in seq if t != pad_id]
        # map to chars
        chars = [categories[t] for t in seq if 0 <= t < len(categories)]
        texts.append("".join(chars))
    return texts


def lm_collate(batch, base_collate_fn, hf_tokenizer, categories, pad_id, max_label_len=128, hybrid=False):
    base = base_collate_fn(batch)
    if len(base) == 5:
        x, y, len_x, len_y, texts = base
    else:
        x, y, len_x, len_y = base
        texts = _chars_to_strings(y, len_y, categories, pad_id)

    tok = hf_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_label_len,
    )

    labels = tok["input_ids"]
    # HF expects -100 for ignore positions
    labels = labels.masked_fill(labels == hf_tokenizer.pad_token_id, -100)

    if hybrid:
        return x, len_x, labels, texts, y, len_y
    return x, len_x, labels, texts


def vlm_collate(batch, base_collate_fn, hf_tokenizer, categories, pad_id, max_label_len=128, hybrid=False):
    """Collate for VLM (decoder-only or seq2seq LM).

    Similar to ``lm_collate`` but appends EOS to the text labels so the model
    learns to produce an end-of-sequence token, and uses attention-mask-based
    padding detection (required when pad_token == eos_token, as in GPT-2).

    For tokenizers that already append EOS automatically (e.g. T5), the manual
    EOS append is skipped to avoid double-EOS corruption.

    Returns:
        ``(x, len_x, labels, texts)`` where *labels* contains token IDs with
        ``-100`` for padding positions.
    """
    import torch

    base = base_collate_fn(batch)
    if len(base) == 5:
        x, y, len_x, len_y, texts = base
    else:
        x, y, len_x, len_y = base
        texts = _chars_to_strings(y, len_y, categories, pad_id)

    # Check if tokenizer automatically appends EOS (e.g. T5) vs not (GPT-2)
    _probe = hf_tokenizer.encode("a")
    tokenizer_adds_eos = (
        len(_probe) > 0 and _probe[-1] == hf_tokenizer.eos_token_id
    )

    if tokenizer_adds_eos:
        # T5-family: tokenizer already appends </s>
        texts_with_eos = texts
    else:
        # GPT-2-family: manually append EOS token
        eos_token = hf_tokenizer.eos_token or ""
        texts_with_eos = [t + eos_token for t in texts]

    tok = hf_tokenizer(
        texts_with_eos,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_label_len,
    )

    labels = tok["input_ids"].clone()
    attn = tok["attention_mask"]  # 1=real, 0=pad

    # Use attention mask to determine padding (correct even when pad==eos)
    labels[attn == 0] = -100

    if hybrid:
        return x, len_x, labels, texts, y, len_y
    return x, len_x, labels, texts
