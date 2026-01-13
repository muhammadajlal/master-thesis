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

def lm_collate(batch, base_collate_fn, hf_tokenizer, categories, pad_id, max_label_len=128):
    x, y, len_x, len_y = base_collate_fn(batch)

    # y is token IDs in "char categories space". Convert to python strings.
    # Assumption: y uses pad_id for padding.
    texts = []
    for i in range(y.size(0)):
        seq = y[i, :len_y[i]].tolist()
        # drop pad ids
        seq = [t for t in seq if t != pad_id]
        # map to chars
        chars = [categories[t] for t in seq if 0 <= t < len(categories)]
        texts.append("".join(chars))

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

    return x, len_x, labels, texts
