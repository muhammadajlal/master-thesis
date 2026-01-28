"""
BPE tokenizer using SentencePiece.
"""

from typing import List

import sentencepiece as spm

from rewi.tokenizer.base import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    """
    BPE tokenizer backed by SentencePiece.
    
    Expected special token IDs (standard SentencePiece configuration):
    - PAD: 0
    - BOS: 1
    - EOS: 2
    - UNK: 3
    
    Example:
        >>> tok = BPETokenizer("path/to/model.model")
        >>> ids = tok.encode("hello world")
        >>> text = tok.decode(ids)
    """
    
    def __init__(self, model_path: str):
        """
        Initialize BPE tokenizer from a trained SentencePiece model.
        
        Args:
            model_path: Path to .model file produced by SentencePiece training.
        """
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self._PAD = self.sp.pad_id()   # 0
        self._BOS = self.sp.bos_id()   # 1
        self._EOS = self.sp.eos_id()   # 2
        self._UNK = self.sp.unk_id()   # 3
        self._vocab_size = self.sp.vocab_size()

    @property
    def PAD(self) -> int:
        return self._PAD
    
    @property
    def BOS(self) -> int:
        return self._BOS
    
    @property
    def EOS(self) -> int:
        return self._EOS
    
    @property
    def UNK(self) -> int:
        return self._UNK
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str) -> List[int]:
        """
        Encode text to BPE token IDs.
        
        Does NOT add BOS/EOS automatically (handled by collate/training).
        """
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: List[int]) -> str:
        """
        Decode BPE token IDs to text.
        
        Automatically removes PAD, BOS, EOS tokens.
        """
        # Remove special tokens
        ids = [t for t in ids if t not in (self._PAD, self._BOS, self._EOS)]
        return self.sp.decode(ids)

    def pieces(self, ids: List[int]) -> List[str]:
        """
        Get subword pieces for token IDs (useful for debugging).
        """
        return self.sp.id_to_piece(ids)
