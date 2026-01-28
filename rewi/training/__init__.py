"""
Training module for handwriting recognition.

This module provides:
- Training loop functions for different model types (CTC, AR, LM)
- Evaluation/testing functions
- Training utilities (freeze/unfreeze, optimizer coverage, etc.)
"""

from rewi.training.loops import (
    train_one_epoch,
    train_one_epoch_lm,
    test,
    test_lm,
)
from rewi.training.utils import (
    build_ar_batch,
    count_params,
    maybe_log_trainability,
    set_decoder_frozen,
    log_decoder_pretrain_load,
    maybe_log_optimizer_coverage,
)

__all__ = [
    # Training loops
    "train_one_epoch",
    "train_one_epoch_lm",
    "test",
    "test_lm",
    # Utilities
    "build_ar_batch",
    "count_params",
    "maybe_log_trainability",
    "set_decoder_frozen",
    "log_decoder_pretrain_load",
    "maybe_log_optimizer_coverage",
]
