"""
Tokenizer module for handwriting recognition.

This module provides:
- BPETokenizer: SentencePiece-based BPE tokenizer
- CharTokenizer: Character-level tokenizer with special tokens
- Utilities for building tokenizer vocabularies

All tokenizers follow a consistent interface:
- encode(text) -> list[int]
- decode(ids) -> str
- PAD, BOS, EOS, UNK special token IDs
- vocab_size property
"""

from rewi.tokenizer.base import BaseTokenizer
from rewi.tokenizer.bpe import BPETokenizer
from rewi.tokenizer.char import CharTokenizer
from rewi.tokenizer.utils import normalize_text

__all__ = [
    "BaseTokenizer",
    "BPETokenizer", 
    "CharTokenizer",
    "normalize_text",
]
