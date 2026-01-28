"""
Base tokenizer interface.

All tokenizers should inherit from this base class to ensure consistent API.
"""

from abc import ABC, abstractmethod
from typing import List


class BaseTokenizer(ABC):
    """
    Abstract base class for tokenizers.
    
    All tokenizers must implement:
    - encode(): Convert text to token IDs
    - decode(): Convert token IDs to text
    
    And provide these properties:
    - PAD, BOS, EOS, UNK: Special token IDs
    - vocab_size: Total vocabulary size
    """
    
    @property
    @abstractmethod
    def PAD(self) -> int:
        """Padding token ID."""
        pass
    
    @property
    @abstractmethod
    def BOS(self) -> int:
        """Beginning of sequence token ID."""
        pass
    
    @property
    @abstractmethod
    def EOS(self) -> int:
        """End of sequence token ID."""
        pass
    
    @property
    @abstractmethod
    def UNK(self) -> int:
        """Unknown token ID."""
        pass
    
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        pass
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """
        Convert text to a list of token IDs.
        
        Note: Does NOT automatically add BOS/EOS tokens.
        """
        pass
    
    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """
        Convert token IDs back to text.
        
        Special tokens (PAD, BOS, EOS) are automatically removed.
        """
        pass
