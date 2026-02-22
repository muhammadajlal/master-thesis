"""
External character-level language model for CTC and AR decoding.

Wraps a KenLM n-gram model to provide:
- score(sequence)     → full-sequence log-probability
- step(prefix, token) → incremental log-probability for next token

The LM operates at the **character** level (matching HWR output).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

import kenlm


class CharLM:
    """Character-level language model backed by KenLM.

    KenLM operates on whitespace-separated tokens. Since our recogniser
    outputs *characters*, each character is treated as a "word" by KenLM.
    A text string ``"hello"`` is fed as ``"h e l l o"``.

    Parameters
    ----------
    model_path : str | Path
        Path to a KenLM ``.arpa`` or ``.binary`` model file.
    """

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = str(model_path)
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"KenLM model not found: {self.model_path}")
        self.model = kenlm.Model(self.model_path)
        self.order = self.model.order

        # Build a KenLM State at <s> (BOS) for incremental scoring
        self._bos_state = kenlm.State()
        self.model.BeginSentenceWrite(self._bos_state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _chars_to_kenlm(text: str) -> str:
        """Convert a character string into KenLM-friendly space-separated tokens.

        Spaces in the original text are mapped to a special ``<space>``
        token (which must also appear in the ARPA training corpus).
        """
        tokens = []
        for ch in text:
            if ch == " ":
                tokens.append("<space>")
            else:
                tokens.append(ch)
        return " ".join(tokens)

    def score(self, text: str) -> float:
        """Return the full-sequence log₁₀ probability of *text*.

        The result is a **log₁₀** value (KenLM default).  Multiply by
        ``math.log(10)`` to convert to natural log if needed.
        """
        kenlm_text = self._chars_to_kenlm(text)
        return self.model.score(kenlm_text, bos=True, eos=True)

    def score_ln(self, text: str) -> float:
        """Return the full-sequence **natural-log** probability."""
        return self.score(text) * math.log(10)

    def step(
        self,
        state: Optional["kenlm.State"],
        token: str,
    ) -> tuple[float, "kenlm.State"]:
        """Incremental scoring: extend *state* by one character *token*.

        Parameters
        ----------
        state : kenlm.State or None
            Current LM state.  Pass ``None`` to start from BOS.
        token : str
            Single character to append (use ``"<space>"`` for space).

        Returns
        -------
        logprob : float
            Log₁₀ probability of *token* given *state*.
        new_state : kenlm.State
            Updated LM state (pass to next call).
        """
        if state is None:
            state = kenlm.State()
            self.model.BeginSentenceWrite(state)

        if token == " ":
            token = "<space>"

        out_state = kenlm.State()
        logprob = self.model.BaseScore(state, token, out_state)
        return logprob, out_state

    def step_ln(
        self,
        state: Optional["kenlm.State"],
        token: str,
    ) -> tuple[float, "kenlm.State"]:
        """Like :meth:`step` but returns **natural-log** probability."""
        lp, s = self.step(state, token)
        return lp * math.log(10), s

    def eos_score(self, state: "kenlm.State") -> float:
        """Score the ``</s>`` (EOS) transition from *state* (log₁₀)."""
        out_state = kenlm.State()
        logprob = self.model.BaseScore(state, "</s>", out_state)
        return logprob

    def eos_score_ln(self, state: "kenlm.State") -> float:
        """Score the ``</s>`` (EOS) transition in natural log."""
        return self.eos_score(state) * math.log(10)

    def bos_state(self) -> "kenlm.State":
        """Return a fresh BOS state."""
        s = kenlm.State()
        self.model.BeginSentenceWrite(s)
        return s


def load_lm(path: str | Path) -> CharLM:
    """Convenience function to load a character LM."""
    return CharLM(path)
