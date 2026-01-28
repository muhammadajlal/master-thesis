"""
DEPRECATED: This module is maintained for backward compatibility.
Please use `from rewi.tokenizer import BPETokenizer, CharTokenizer` instead.
"""

# Re-export from new module structure for backward compatibility
from rewi.tokenizer.bpe import BPETokenizer
from rewi.tokenizer.char import CharTokenizer
from rewi.tokenizer.utils import normalize_text as _norm_text

__all__ = ["BPETokenizer", "CharTokenizer", "_norm_text"]
