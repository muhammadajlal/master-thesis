"""
Neural language model wrapper for AR decoding.

Wraps a pretrained ``ARDecoder`` (text-only) to provide the same API as
:class:`rewi.decoding.lm.CharLM`, so it can be used as a drop-in LM for:

- N-best rescoring  (``score_ln(text)``)
- Shallow fusion    (``step_ln(state, token) → (logprob, state)``)

The pretrained decoder was trained text-only with a constant-zero memory
tensor (see ``pretrain_decoder.py``), so we use the same dummy memory here.

Usage
-----
::

    from rewi.decoding.neural_lm import NeuralCharLM, load_neural_lm

    nlm = load_neural_lm(
        checkpoint="results/.../best_loss.pth",
        categories=["", "A", "B", ...],  # same as training
    )
    # Full-sequence scoring (natural log)
    lp = nlm.score_ln("hello")

    # Step-level for shallow fusion
    state = nlm.bos_state()
    lp, state = nlm.step_ln(state, "h")
    lp, state = nlm.step_ln(state, "e")
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from rewi.model.ARDecoder import ARDecoder


# ──────────────────────────────────────────────────────────────────────
# State object for incremental decoding
# ──────────────────────────────────────────────────────────────────────

@dataclass
class NeuralLMState:
    """State carried between :meth:`NeuralCharLM.step_ln` calls.

    Stores the token-ID prefix so far (excluding BOS, which is prepended
    automatically by the scorer).  This is the "state" in the beam search
    analogy — effectively the entire decoding history.
    """
    ids: list[int] = field(default_factory=list)

    def copy(self) -> "NeuralLMState":
        """Cheap copy for beam branching."""
        return NeuralLMState(ids=list(self.ids))


# ──────────────────────────────────────────────────────────────────────
# Neural character-level LM
# ──────────────────────────────────────────────────────────────────────

class NeuralCharLM:
    """Character-level neural language model backed by a pretrained ARDecoder.

    Duck-types the :class:`rewi.decoding.lm.CharLM` interface so it works
    as a drop-in replacement in ``ar_beam.py`` and ``ctc_beam.py``.

    Parameters
    ----------
    decoder : ARDecoder
        Pretrained decoder (already loaded with weights).
    categories : list[str]
        Character list (index 0 = blank / empty string).  Must match the
        vocabulary used during pretraining.
    device : str | torch.device
        Device to run inference on.
    """

    def __init__(
        self,
        decoder: ARDecoder,
        categories: list[str],
        device: str | torch.device = "cpu",
    ) -> None:
        self.decoder = decoder
        self.decoder.eval()
        self.categories = categories
        self.device = torch.device(device)
        self.decoder = self.decoder.to(self.device)

        # Build char → token-ID mapping (skip blank at index 0)
        self._char2id: dict[str, int] = {}
        for idx, ch in enumerate(categories):
            if ch:  # skip blank
                self._char2id[ch] = idx

        # Special token IDs (must match pretraining convention)
        base = len(categories)
        self.PAD_ID = base
        self.BOS_ID = base + 1
        self.EOS_ID = base + 2
        self.vocab_size = base + 3

        # Dummy memory (constant zero, as used during pretraining)
        self._dummy_mem = torch.zeros(
            (1, 1, self.decoder.d_model),
            device=self.device, dtype=torch.float32,
        )
        self._dummy_pad = torch.zeros(
            (1, 1), device=self.device, dtype=torch.bool,
        )

        # Friendly name
        self.order = "neural"

    # ------------------------------------------------------------------
    # Text ↔ token-ID helpers
    # ------------------------------------------------------------------

    def _text_to_ids(self, text: str) -> list[int]:
        """Convert a character string to a list of token IDs."""
        ids = []
        for ch in text:
            tid = self._char2id.get(ch)
            if tid is not None:
                ids.append(tid)
            # Unknown chars are silently dropped (like pretraining)
        return ids

    # ------------------------------------------------------------------
    # Core scoring functions
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _forward_ids(self, ids: list[int]) -> torch.Tensor:
        """Run the decoder on [BOS] + ids and return log-probs ``(T, V)``.

        Returns the log-softmax distribution at each position, so
        ``result[t]`` is ``log P(token | ids[:t])``.
        """
        inp = [self.BOS_ID] + ids
        y = torch.tensor([inp], dtype=torch.long, device=self.device)
        logits = self.decoder(y, self._dummy_mem, self._dummy_pad)  # (1, T, V)
        return F.log_softmax(logits[0].float(), dim=-1)  # (T, V)

    # ------------------------------------------------------------------
    # Public API  (mirrors CharLM)
    # ------------------------------------------------------------------

    def score(self, text: str) -> float:
        """Full-sequence log₁₀ probability of *text*.

        Sums ``log P(c_t | c_{<t})`` for each character plus ``log P(EOS)``.
        """
        return self.score_ln(text) / math.log(10)

    def score_ln(self, text: str) -> float:
        """Full-sequence **natural-log** probability."""
        ids = self._text_to_ids(text)
        if not ids:
            return 0.0
        lp = self._forward_ids(ids)  # (T, V), T = 1 + len(ids)
        # lp[0] = P(first char | BOS), lp[1] = P(second char | first), ...
        total = 0.0
        for t, tid in enumerate(ids):
            total += float(lp[t, tid])
        # Add EOS score
        total += float(lp[len(ids), self.EOS_ID])
        return total

    def step(
        self,
        state: Optional[NeuralLMState],
        token: str,
    ) -> tuple[float, NeuralLMState]:
        """Incremental scoring: extend *state* by one character *token*.

        Returns ``(log₁₀_prob, new_state)``.
        """
        lp, s = self.step_ln(state, token)
        return lp / math.log(10), s

    def step_ln(
        self,
        state: Optional[NeuralLMState],
        token: str,
    ) -> tuple[float, NeuralLMState]:
        """Incremental scoring returning **natural-log** probability.

        Parameters
        ----------
        state : NeuralLMState or None
            Current state.  Pass ``None`` to start from BOS.
        token : str
            Single character to extend.

        Returns
        -------
        logprob : float
            Natural-log probability of *token* given the prefix.
        new_state : NeuralLMState
            Updated state (pass to next call).
        """
        if state is None:
            state = NeuralLMState(ids=[])

        # Map to token ID
        if token == " ":
            token = " "  # keep as-is (no <space> mapping for neural LM)
        tid = self._char2id.get(token)
        if tid is None:
            # Unknown character — return a small penalty, don't advance state
            return -10.0, state.copy()

        # Run decoder on prefix so far
        lp = self._forward_ids(state.ids)  # (T, V)
        # Score at position len(state.ids) corresponds to P(next | prefix)
        log_prob = float(lp[len(state.ids), tid])

        new_state = state.copy()
        new_state.ids.append(tid)
        return log_prob, new_state

    def eos_score(self, state: NeuralLMState) -> float:
        """Score the EOS transition from *state* (log₁₀)."""
        return self.eos_score_ln(state) / math.log(10)

    def eos_score_ln(self, state: NeuralLMState) -> float:
        """Score the EOS transition from *state* (natural log)."""
        if state is None:
            state = NeuralLMState(ids=[])
        lp = self._forward_ids(state.ids)  # (T, V)
        return float(lp[len(state.ids), self.EOS_ID])

    def bos_state(self) -> NeuralLMState:
        """Return a fresh BOS state."""
        return NeuralLMState(ids=[])


# ──────────────────────────────────────────────────────────────────────
# Key migration (bare decoder → new GatedMHA format)
# ──────────────────────────────────────────────────────────────────────

def _migrate_decoder_keys(state_dict: dict) -> dict:
    """Migrate old Q/K/V projection keys to nn.MultiheadAttention format.

    Pretrained checkpoints save bare ARDecoder state dicts with separate
    ``q_proj``, ``k_proj``, ``v_proj`` keys.  Modern GatedTransformerDecoderLayer
    stores them inside ``mha.in_proj_weight``.
    """
    import re
    new_sd = {}
    consumed = set()
    old_prefixes = set()

    for k in state_dict:
        m = re.match(r"^(.+\.(self_attn|multihead_attn))\.q_proj\.weight$", k)
        if m:
            old_prefixes.add(m.group(1))

    for prefix in old_prefixes:
        q_w = state_dict.get(f"{prefix}.q_proj.weight")
        k_w = state_dict.get(f"{prefix}.k_proj.weight")
        v_w = state_dict.get(f"{prefix}.v_proj.weight")
        q_b = state_dict.get(f"{prefix}.q_proj.bias")
        k_b = state_dict.get(f"{prefix}.k_proj.bias")
        v_b = state_dict.get(f"{prefix}.v_proj.bias")
        o_w = state_dict.get(f"{prefix}.out_proj.weight")
        o_b = state_dict.get(f"{prefix}.out_proj.bias")

        if q_w is not None and k_w is not None and v_w is not None:
            new_sd[f"{prefix}.mha.in_proj_weight"] = torch.cat([q_w, k_w, v_w], dim=0)
            consumed.update([f"{prefix}.{p}.weight" for p in ("q_proj", "k_proj", "v_proj")])
            if q_b is not None and k_b is not None and v_b is not None:
                new_sd[f"{prefix}.mha.in_proj_bias"] = torch.cat([q_b, k_b, v_b], dim=0)
                consumed.update([f"{prefix}.{p}.bias" for p in ("q_proj", "k_proj", "v_proj")])
        if o_w is not None:
            new_sd[f"{prefix}.mha.out_proj.weight"] = o_w
            consumed.add(f"{prefix}.out_proj.weight")
        if o_b is not None:
            new_sd[f"{prefix}.mha.out_proj.bias"] = o_b
            consumed.add(f"{prefix}.out_proj.bias")

    for k, v in state_dict.items():
        if k not in consumed:
            new_sd[k] = v
    return new_sd


# ──────────────────────────────────────────────────────────────────────
# Convenience loader
# ──────────────────────────────────────────────────────────────────────

def load_neural_lm(
    checkpoint: str | Path,
    categories: list[str],
    *,
    device: str = "cpu",
    d_model: int = 256,
    nhead: int = 4,
    layers: int = 4,
    dim_ff: int = 1024,
    use_gated_attention: bool = True,
    gating_type: str = "elementwise",
) -> NeuralCharLM:
    """Load a pretrained ARDecoder as a neural character LM.

    Parameters
    ----------
    checkpoint : str | Path
        Path to the ``best_loss.pth`` (or any checkpoint from
        ``pretrain_decoder.py``).
    categories : list[str]
        Character list (with blank at index 0).
    device : str
        Device for inference.
    d_model, nhead, layers, dim_ff : int
        Architecture parameters (must match pretraining config).
    use_gated_attention : bool
        Whether gated attention was used during pretraining.
    gating_type : str
        Type of gating (``"elementwise"`` or ``"headwise"``).

    Returns
    -------
    NeuralCharLM
        Neural LM wrapper ready for use in beam search.
    """
    checkpoint = str(checkpoint)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Neural LM checkpoint not found: {checkpoint}")

    # Build decoder with matching architecture
    base = len(categories)
    vocab_size = base + 3  # +PAD, +BOS, +EOS

    decoder = ARDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        layers=layers,
        dim_ff=dim_ff,
        use_gated_attention=use_gated_attention,
        gating_type=gating_type,
    )

    # Load checkpoint
    ckp = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckp.get("model", ckp)
    state = _migrate_decoder_keys(state)

    res = decoder.load_state_dict(state, strict=False)
    if res.missing_keys:
        print(f"[NeuralCharLM] missing keys: {res.missing_keys}")
    if res.unexpected_keys:
        print(f"[NeuralCharLM] unexpected keys: {res.unexpected_keys}")

    return NeuralCharLM(decoder, categories, device=device)
