# rewi/dataset_concat.py
import random
import torch
from collections import defaultdict


class ConcatWordDataset(torch.utils.data.Dataset):
    """
    Wrap an existing word-level dataset. Each __getitem__ returns a single
    concatenated sample built from K original items (K in [items_min, items_max]),
    with an optional max_T "budget" (stop concatenation before exceeding max_T).

    Key improvements:
      - same-writer concatenation using JSON key: "id_writer"
      - avoid truncating x without truncating y (use budget instead)
      - optional mixing (probability_concat) to keep original distribution
      - optional disabling of base augmentations inside concatenation
    """
    def __init__(
        self,
        base_ds,
        items_min=2,
        items_max=4,
        max_T=None,
        use_separator=False,
        sep_id=None,
        pad_id=None,
        same_writer=False,
        writer_key="id_writer",          # <-- UPDATED DEFAULT
        p_concat=1.0,
        disable_base_aug_in_concat=True,
        debug_return_meta=False,
        debug_assert_same_writer=False,
    ):
        self.base = base_ds
        self.items_min = items_min
        self.items_max = max(items_min, items_max)
        self.max_T = max_T
        self.use_separator = use_separator
        self.sep_id = sep_id      # int or None (decoder-vocab id)
        self.pad_id = pad_id      # int or None

        self.same_writer = same_writer
        self.writer_key = writer_key
        self.p_concat = float(p_concat)
        self.disable_base_aug_in_concat = disable_base_aug_in_concat
        self.debug_return_meta = debug_return_meta
        self.debug_assert_same_writer = debug_assert_same_writer


        # Precompute writer -> indices map if available
        self.writer_to_indices = None
        if self.same_writer and hasattr(self.base, "annos"):
            m = defaultdict(list)
            for i, a in enumerate(self.base.annos):
                w = a.get(self.writer_key, None)
                m[w].append(i)
            self.writer_to_indices = dict(m)

    def __len__(self):
        return len(self.base)

    def _fetch(self, idx):
        sample = self.base[idx]
        if isinstance(sample, (list, tuple)) and len(sample) == 4:
            x, y, lx, ly = sample
            x = torch.as_tensor(x)
            y = torch.as_tensor(y)[:int(ly)]
        else:
            x, y = sample
            x = torch.as_tensor(x)
            y = torch.as_tensor(y)

        # Strip labels at PAD if provided
        if self.pad_id is not None:
            pad_pos = (y == self.pad_id).nonzero(as_tuple=True)[0]
            if pad_pos.numel():
                y = y[:pad_pos[0]]

        # Infer len_x (time dim is the larger one)
        lx = max(x.shape)

        # Normalize x as (C, T)
        if x.dim() != 2:
            raise ValueError(f"Expected 2D (C,T)/(T,C); got {tuple(x.shape)}")
        if x.size(0) > x.size(1) and x.size(1) <= 64:
            x = x.t().contiguous()

        return x, y, int(lx), int(y.numel())

    def _sample_indices(self, idx, K):
        # Fallback: sequential sampling
        if not (self.same_writer and self.writer_to_indices is not None and hasattr(self.base, "annos")):
            return [(idx + j) % len(self.base) for j in range(K)]

        w = self.base.annos[idx].get(self.writer_key, None)
        pool = self.writer_to_indices.get(w, None)

        if not pool:
            return [(idx + j) % len(self.base) for j in range(K)]

        # Ensure idx is included; sample the rest from same writer
        others = [i for i in pool if i != idx]
        if len(others) >= (K - 1):
            chosen = random.sample(others, K - 1)
        else:
            # If writer has too few samples, sample with replacement
            chosen = [random.choice(pool) for _ in range(K - 1)]

        inds = [idx] + chosen
        random.shuffle(inds)
        return inds

    def __getitem__(self, idx):
        # Optional mixing: sometimes return original sample without concatenation
        if self.p_concat < 1.0 and random.random() > self.p_concat:
            x, y = self.base[idx]
            x = torch.as_tensor(x)
            if x.dim() != 2:
                raise ValueError(f"Expected 2D for x; got {tuple(x.shape)}")
            if x.size(0) > x.size(1) and x.size(1) <= 64:
                x = x.t().contiguous()
            y = torch.as_tensor(y)
            if self.debug_return_meta and hasattr(self.base, "annos"):
                w = self.base.annos[idx].get(self.writer_key, None)
                return x, y, x.size(1), y.numel(), [idx], [w]
            return x, y, x.size(1), y.numel()

        # Temporarily disable base augmentations during concatenation (optional but recommended)
        old_augs = None
        if self.disable_base_aug_in_concat and hasattr(self.base, "augs"):
            old_augs = self.base.augs
            self.base.augs = None

        try:
            K = random.randint(self.items_min, self.items_max)
            inds = self._sample_indices(idx, K)
            writers = None
            if hasattr(self.base, "annos"):
                writers = [self.base.annos[i].get(self.writer_key, None) for i in inds]

            if self.same_writer and self.debug_assert_same_writer and writers is not None:
                # allow None writers to pass if your json is incomplete; otherwise assert strict equality
                if any(w is None for w in writers):
                    pass
                else:
                    w0 = writers[0]
                    assert all(w == w0 for w in writers), f"Writer mismatch: {writers} for inds={inds}"


            xs, ys = [], []
            tot_T = 0

            for i in inds:
                x_i, y_i, lx_i, _ = self._fetch(i)

                # max_T as a "budget": do not truncate x without matching y
                if self.max_T is not None and (tot_T + lx_i) > self.max_T:
                    if len(xs) == 0:
                        # If first sample alone exceeds max_T, keep it (no truncation)
                        pass
                    else:
                        break

                xs.append(x_i[:, :lx_i])
                if self.use_separator and self.sep_id is not None and len(ys) > 0:
                    ys.append(torch.tensor([self.sep_id], dtype=torch.long))
                ys.append(y_i)

                tot_T += lx_i

            x_cat = torch.cat(xs, dim=1)  # (C, sum_T)
            y_cat = torch.cat(ys, dim=0)  # (S_total,)

            if self.debug_return_meta:
                return x_cat, y_cat, x_cat.size(1), y_cat.numel(), inds, writers
            return x_cat, y_cat, x_cat.size(1), y_cat.numel()


        finally:
            if old_augs is not None:
                self.base.augs = old_augs


def concat_collate(batch, pad_value_x=0.0, pad_value_y=None):
    # batch: list of (x_cat, y_cat, len_x, len_y)
    C = batch[0][0].size(0)
    T_max = max(item[2] for item in batch)
    S_max = max(item[3] for item in batch)
    B = len(batch)

    x_batch = torch.full((B, C, T_max), pad_value_x, dtype=batch[0][0].dtype)
    y_batch = torch.full(
        (B, S_max),
        pad_value_y if pad_value_y is not None else 0,
        dtype=torch.long,
    )
    len_x = torch.empty(B, dtype=torch.long)
    len_y = torch.empty(B, dtype=torch.long)

    for b, (x, y, lx, ly) in enumerate(batch):
        x_batch[b, :, :lx] = x
        y_batch[b, :ly] = y
        len_x[b] = lx
        len_y[b] = ly

    return x_batch, y_batch, len_x, len_y
