"""
AR decoding: greedy, calibrated beam search, N-best rescoring, shallow fusion.

Calibrated beam search scoring (base):
    S(y) = (1 / |y|^α) · log P_AR(y|x)  +  γ · 1[EOS]  (with min-length constraint)

N-best rescoring:
    S(y) = β_ar · log P_AR(y|x) / |y|^α  +  λ · log P_LM(y)  +  β · |y|

Shallow fusion (inside beam):
    step score = log P_AR(next|prefix, x) + λ · log P_LM(next|prefix)

All functions are per-utterance for clarity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Callable

import torch
import torch.nn.functional as F

from rewi.decoding.lm import CharLM

# Type alias: any object with the CharLM API (CharLM or NeuralCharLM)
LMType = CharLM  # duck-typed; NeuralCharLM also accepted


# ──────────────────────────────────────────────────────────────────────
# 1.  AR greedy  (with metadata for evaluation harness)
# ──────────────────────────────────────────────────────────────────────


@torch.no_grad()
def ar_greedy(
    memory: torch.Tensor,          # (1, Tm, D)
    enc_pad: torch.Tensor,         # (1, Tm)
    decoder_fn: Callable,          # decoder(y_inp, memory, enc_pad) → logits
    *,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_len: int = 200,
    categories: list[str] | None = None,
    tokenizer=None,
) -> dict:
    """AR greedy decoding with full metadata.

    Returns
    -------
    dict with keys:
        text :  str
        ids  :  list[int]  (no BOS/EOS/PAD)
        ended_by_eos : bool
        log_prob : float   (sum of step log-probs)
        length : int
    """
    device = memory.device
    y = torch.tensor([[bos_id]], dtype=torch.long, device=device)
    ids: list[int] = []
    total_lp = 0.0
    ended_by_eos = False

    for _ in range(max_len):
        logits = decoder_fn(y, memory, enc_pad)  # (1, t, V)
        lp = F.log_softmax(logits[:, -1, :], dim=-1)  # (1, V)
        best = lp.argmax(dim=-1).item()
        total_lp += float(lp[0, best])

        if best == eos_id:
            ended_by_eos = True
            break
        if best == pad_id:
            continue
        ids.append(best)
        y = torch.cat([y, torch.tensor([[best]], device=device)], dim=1)

    text = _decode_ids(ids, categories, tokenizer)
    return {
        "text": text,
        "ids": ids,
        "ended_by_eos": ended_by_eos,
        "log_prob": total_lp,
        "length": len(ids),
    }


# ──────────────────────────────────────────────────────────────────────
# 2.  AR calibrated beam search
# ──────────────────────────────────────────────────────────────────────


@dataclass
class _ARBeam:
    """State for one hypothesis in AR beam search."""
    ids: list[int] = field(default_factory=list)       # decoded IDs (no BOS)
    log_prob: float = 0.0
    ended: bool = False
    lm_state: object = None
    lm_score: float = 0.0

    def score(self, alpha: float, eos_bias: float = 0.0) -> float:
        """Length-normalised score with optional EOS bias."""
        L = max(1, len(self.ids))
        s = self.log_prob / (L ** alpha) if alpha > 0 else self.log_prob
        if self.ended:
            s += eos_bias  # bias once upon completion
        return s


@torch.no_grad()
def ar_calibrated_beam_search(
    memory: torch.Tensor,          # (1, Tm, D)
    enc_pad: torch.Tensor,         # (1, Tm)
    decoder_fn: Callable,
    *,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    beam_size: int = 4,
    max_len: int = 200,
    alpha: float = 0.6,            # length normalisation exponent
    eos_bias: float = 0.0,         # additive bias on EOS logit (γ)
    min_len: int = 0,              # minimum length before EOS is allowed
    min_len_ratio: float = 0.0,    # min length as ratio of encoder length
    categories: list[str] | None = None,
    tokenizer=None,
    # ---- LM shallow fusion ----
    lm: Optional[CharLM] = None,
    lm_weight: float = 0.0,
) -> list[dict]:
    """AR calibrated beam search.

    Returns list of N-best hypotheses (best first), each dict:
        text, ids, score, log_prob, lm_score, ended_by_eos, length
    """
    device = memory.device

    # Compute effective min_len
    if min_len_ratio > 0:
        Tm = int((~enc_pad[0]).sum().item())
        min_len = max(min_len, int(Tm * min_len_ratio))

    # Initialise beam
    init_beam = _ARBeam(ids=[], log_prob=0.0)
    if lm is not None and lm_weight > 0:
        init_beam.lm_state = lm.bos_state()
    beams: list[_ARBeam] = [init_beam]

    for step in range(max_len):
        all_cands: list[_ARBeam] = []

        for beam in beams:
            if beam.ended:
                all_cands.append(beam)
                continue

            # Build input sequence: [BOS] + ids
            inp = [bos_id] + beam.ids
            y_inp = torch.tensor([inp], dtype=torch.long, device=device)
            logits = decoder_fn(y_inp, memory, enc_pad)  # (1, t, V)
            lp = F.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)  # (V,)

            # Apply EOS bias
            if eos_bias != 0.0:
                lp[eos_id] = lp[eos_id] + eos_bias

            # Mask EOS if below min_len
            if step < min_len:
                lp[eos_id] = -math.inf

            # Top-K expansion
            K = min(beam_size * 2, lp.numel())  # over-generate then prune
            topv, topi = torch.topk(lp, k=K, dim=-1)

            for lp_val, tid in zip(topv.tolist(), topi.tolist()):
                if tid == pad_id:
                    continue
                new_ended = (tid == eos_id)
                new_ids = beam.ids + ([] if new_ended else [tid])
                new_lp = beam.log_prob + lp_val

                # LM shallow fusion
                new_lm_state = beam.lm_state
                new_lm_score = beam.lm_score
                if lm is not None and lm_weight > 0 and not new_ended:
                    ch = _id_to_char(tid, categories, tokenizer)
                    if ch:
                        lm_lp, new_lm_state = lm.step_ln(beam.lm_state, ch)
                        new_lm_score = beam.lm_score + lm_lp

                cand = _ARBeam(
                    ids=new_ids,
                    log_prob=new_lp,
                    ended=new_ended,
                    lm_state=new_lm_state,
                    lm_score=new_lm_score,
                )
                all_cands.append(cand)

        # Prune to beam_size
        def _rank(b: _ARBeam) -> float:
            s = b.score(alpha, eos_bias=0.0)  # don't double-count EOS bias
            if lm is not None and lm_weight > 0:
                s += lm_weight * b.lm_score / max(1, len(b.ids))
            return s

        all_cands.sort(key=_rank, reverse=True)
        beams = all_cands[:beam_size]

        if all(b.ended for b in beams):
            break

    # Finalise
    results = []
    for beam in beams:
        text = _decode_ids(beam.ids, categories, tokenizer)
        score = beam.score(alpha)
        if lm is not None and lm_weight > 0:
            score += lm_weight * beam.lm_score / max(1, len(beam.ids))
        results.append({
            "text": text,
            "ids": beam.ids,
            "score": score,
            "log_prob": beam.log_prob,
            "lm_score": beam.lm_score,
            "ended_by_eos": beam.ended,
            "length": len(beam.ids),
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# ──────────────────────────────────────────────────────────────────────
# 3.  AR N-best rescoring with external LM
# ──────────────────────────────────────────────────────────────────────


def ar_nbest_rescore(
    nbest: list[dict],
    lm: CharLM,
    *,
    lm_weight: float = 0.4,
    length_bonus: float = 0.0,
    alpha: float = 0.6,
) -> list[dict]:
    """Re-rank an N-best list from AR beam search using an external LM.

    Scoring:
        S(y) = log P_AR(y|x) / |y|^α  +  λ · log P_LM(y)  +  β · |y|

    Parameters
    ----------
    nbest : list[dict]
        N-best hypotheses from ``ar_calibrated_beam_search()``.
    lm : CharLM
        External character LM.
    lm_weight : float (λ)
        Weight for external LM score.
    length_bonus : float (β)
        Per-character bonus (positive = encourage longer).
    alpha : float
        Length normalisation exponent for AR score.

    Returns
    -------
    Re-ranked list of dicts (same fields as input + ``rescore_score``).
    """
    for hyp in nbest:
        text = hyp["text"]
        lm_score = lm.score_ln(text)
        L = max(1, hyp["length"])
        ar_score = hyp["log_prob"] / (L ** alpha) if alpha > 0 else hyp["log_prob"]
        combined = ar_score + lm_weight * lm_score + length_bonus * L
        hyp["lm_score"] = lm_score
        hyp["rescore_score"] = combined

    nbest.sort(key=lambda h: h["rescore_score"], reverse=True)
    return nbest


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _decode_ids(
    ids: list[int],
    categories: list[str] | None,
    tokenizer=None,
) -> str:
    if tokenizer is not None:
        return tokenizer.decode(ids)
    if categories is not None:
        return "".join(
            categories[i] for i in ids
            if 0 <= i < len(categories) and i != 0
        )
    return ""


def _id_to_char(
    tid: int,
    categories: list[str] | None,
    tokenizer=None,
) -> str:
    """Convert a single token ID to its character representation."""
    if tokenizer is not None:
        return tokenizer.decode([tid])
    if categories is not None and 0 <= tid < len(categories):
        return categories[tid]
    return ""
