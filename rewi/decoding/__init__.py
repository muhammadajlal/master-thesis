"""
Decoding module for IMU Handwriting Recognition.

Provides:
- External language model integration (KenLM character n-gram)
- CTC prefix beam search with optional LM shallow fusion
- AR calibrated beam search with length normalization, EOS control,
  and external LM integration (rescoring + shallow fusion)
"""

from rewi.decoding.lm import CharLM, load_lm
from rewi.decoding.neural_lm import NeuralCharLM, load_neural_lm
from rewi.decoding.ctc_beam import ctc_greedy, ctc_prefix_beam_search
from rewi.decoding.ar_beam import ar_greedy, ar_calibrated_beam_search, ar_nbest_rescore

__all__ = [
    "CharLM",
    "load_lm",
    "NeuralCharLM",
    "load_neural_lm",
    "ctc_greedy",
    "ctc_prefix_beam_search",
    "ar_greedy",
    "ar_calibrated_beam_search",
    "ar_nbest_rescore",
]
