"""
CTC decoding: greedy and prefix beam search with optional LM shallow fusion.

All functions operate on a single utterance (no batching) to keep the search
logic clean.  The evaluation harness handles batching / GPU transfer.

Scoring formula for CTC prefix beam + LM:
    S(y) = log P_CTC(y|x) + λ · log P_LM(y) + β · |y|

References:
  - Hannun et al., "First-Pass Large Vocabulary Continuous Speech
    Recognition using Bi-Directional Recurrent DNNs", 2014.
  - Graves & Jaitly, "Towards End-To-End Speech Recognition with
    Recurrent Neural Networks", ICML 2014.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from rewi.decoding.lm import CharLM

# Type alias: any object with the CharLM API (CharLM or NeuralCharLM)
LMType = CharLM  # duck-typed; NeuralCharLM also accepted


# ──────────────────────────────────────────────────────────────────────
# 1.  CTC greedy  (drop-in replacement for BestPath with metadata)
# ──────────────────────────────────────────────────────────────────────


def ctc_greedy(
    logits: torch.Tensor,
    categories: list[str],
    blank: int = 0,
) -> dict:
    """CTC greedy (best-path) decoding.

    Parameters
    ----------
    logits : Tensor (T, V)
        Log-probabilities or raw logits over the vocabulary at each frame.
    categories : list[str]
        Character list (index 0 = blank).
    blank : int
        Blank token index.

    Returns
    -------
    dict with keys:
        text : str – decoded text
        ids  : list[int] – character ids (no blanks/repeats)
    """
    ids_raw = logits.argmax(dim=-1).tolist()
    # collapse repeats + remove blanks
    ids: list[int] = []
    prev = -1
    for t in ids_raw:
        if t != prev:
            if t != blank:
                ids.append(t)
            prev = t
        else:
            prev = t

    text = "".join(categories[i] for i in ids if 0 <= i < len(categories))
    return {"text": text, "ids": ids}


# ──────────────────────────────────────────────────────────────────────
# 2.  CTC prefix beam search  (with optional LM shallow fusion)
# ──────────────────────────────────────────────────────────────────────


@dataclass
class _Beam:
    """State for one prefix in CTC prefix beam search."""
    p_blank: float = -math.inf    # log-prob ending in blank
    p_nonblank: float = -math.inf # log-prob ending in non-blank
    lm_state: object = None       # KenLM state (or None)
    lm_score: float = 0.0         # accumulated LM log-prob (natural log)

    @property
    def total(self) -> float:
        return _logadd(self.p_blank, self.p_nonblank)


def _logadd(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def ctc_prefix_beam_search(
    logits: torch.Tensor,
    categories: list[str],
    *,
    beam_size: int = 25,
    blank: int = 0,
    lm: Optional[CharLM] = None,
    lm_weight: float = 0.0,
    insertion_bonus: float = 0.0,
) -> list[dict]:
    """CTC prefix beam search with optional LM shallow fusion.

    Parameters
    ----------
    logits : Tensor (T, V)
        **Log-softmax** output (or raw logits — softmax is applied internally).
    categories : list[str]
        Character list where ``categories[0]`` is blank.
    beam_size : int
        Number of active prefixes to keep.
    blank : int
        Blank index.
    lm : CharLM or None
        External character LM for shallow fusion.
    lm_weight : float (λ)
        Weight applied to LM log-prob.
    insertion_bonus : float (β)
        Per-character bonus (positive = encourage longer outputs).

    Returns
    -------
    list[dict]  — N-best hypotheses sorted by score (best first).
        Each dict: ``text``, ``ids``, ``score``, ``ctc_score``, ``lm_score``.
    """
    T, V = logits.shape

    # Ensure log-probs
    log_probs = F.log_softmax(logits.float(), dim=-1)  # (T, V)
    log_probs = log_probs.cpu()

    # Initialise with empty prefix
    beams: dict[tuple[int, ...], _Beam] = defaultdict(_Beam)
    empty = ()
    beams[empty].p_blank = 0.0  # log(1)
    if lm is not None:
        beams[empty].lm_state = lm.bos_state()

    for t in range(T):
        lp = log_probs[t]  # (V,)
        new_beams: dict[tuple[int, ...], _Beam] = defaultdict(_Beam)

        # Prune: keep top-K prefixes by total prob
        scored = sorted(
            beams.items(),
            key=lambda kv: kv[1].total,
            reverse=True,
        )[:beam_size]

        for prefix, beam in scored:
            # --- extend with blank ---
            nb = new_beams[prefix]
            p_ext = beam.total + float(lp[blank])
            nb.p_blank = _logadd(nb.p_blank, p_ext)
            # Carry LM state forward (no new char)
            if nb.lm_state is None and beam.lm_state is not None:
                nb.lm_state = beam.lm_state
                nb.lm_score = beam.lm_score

            # --- extend with each non-blank character ---
            for c in range(V):
                if c == blank:
                    continue

                if len(prefix) > 0 and prefix[-1] == c:
                    # Same as last char: can only extend through blank
                    p_ext = beam.p_blank + float(lp[c])
                    new_prefix = prefix + (c,)
                else:
                    p_ext = beam.total + float(lp[c])
                    new_prefix = prefix + (c,)

                nb2 = new_beams[new_prefix]

                # LM scoring for new character
                if lm is not None and lm_weight > 0:
                    ch = categories[c] if 0 <= c < len(categories) else ""
                    if ch and nb2.lm_state is None:
                        lm_lp, lm_st = lm.step_ln(beam.lm_state, ch)
                        nb2.lm_state = lm_st
                        nb2.lm_score = beam.lm_score + lm_lp

                nb2.p_nonblank = _logadd(nb2.p_nonblank, p_ext)

        beams = new_beams

    # Build N-best list
    results = []
    for prefix, beam in beams.items():
        ctc_score = beam.total
        lm_score_val = beam.lm_score if lm is not None else 0.0
        length = len(prefix)
        combined = ctc_score + lm_weight * lm_score_val + insertion_bonus * length
        text = "".join(
            categories[c] for c in prefix if 0 <= c < len(categories)
        )
        results.append({
            "text": text,
            "ids": list(prefix),
            "score": combined,
            "ctc_score": ctc_score,
            "lm_score": lm_score_val,
            "length": length,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results
