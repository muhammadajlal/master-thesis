"""
Character-level tokenizer.
"""

import json
from pathlib import Path
from typing import List, Optional, Union

from rewi.tokenizer.base import BaseTokenizer
from rewi.tokenizer.utils import normalize_text


class CharTokenizer(BaseTokenizer):
    """
    Minimal character tokenizer with special tokens.

    - Encodes a string into character IDs (no automatic BOS/EOS).
    - Decodes IDs back into a string (special tokens removed).
    - Can save/load vocab JSON for reproducible downstream use.
    
    Special tokens are always first in vocabulary:
    - PAD: 0
    - BOS: 1
    - EOS: 2
    - UNK: 3
    
    Example:
        >>> tok = CharTokenizer.build_from_text_file("corpus.txt")
        >>> tok.save("vocab.json")
        >>> tok = CharTokenizer.load("vocab.json")
    """

    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(
        self,
        vocab: List[str],
        *,
        lowercase: bool = True,
        nfkc: bool = True,
    ) -> None:
        """
        Initialize character tokenizer with vocabulary.
        
        Args:
            vocab: List of characters (excluding special tokens, which are added).
            lowercase: Whether to lowercase text during encoding.
            nfkc: Whether to apply NFKC normalization during encoding.
        """
        self.lowercase = lowercase
        self.nfkc = nfkc

        # Ensure specials exist and are first (stable IDs)
        specials = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        seen = set()
        ordered: List[str] = []
        for tok in specials + vocab:
            if tok in seen:
                continue
            seen.add(tok)
            ordered.append(tok)

        self.itos = ordered
        self.stoi = {t: i for i, t in enumerate(self.itos)}

        self._PAD = self.stoi[self.PAD_TOKEN]
        self._BOS = self.stoi[self.BOS_TOKEN]
        self._EOS = self.stoi[self.EOS_TOKEN]
        self._UNK = self.stoi[self.UNK_TOKEN]
        self._vocab_size = len(self.itos)

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

    @classmethod
    def build_from_text_file(
        cls,
        path: Union[str, Path],
        *,
        lowercase: bool = True,
        nfkc: bool = True,
        max_lines: Optional[int] = None,
    ) -> "CharTokenizer":
        """
        Build tokenizer vocabulary from a text file.
        
        Each line is treated as one text sample. Characters are extracted
        and deduplicated to form the vocabulary.
        
        Args:
            path: Path to text file.
            lowercase: Whether to lowercase during vocab building.
            nfkc: Whether to NFKC normalize during vocab building.
            max_lines: Optional limit on lines to read.
        
        Returns:
            CharTokenizer instance with vocabulary from file.
        """
        path = Path(path)
        chars: set = set()
        
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if max_lines is not None and i >= max_lines:
                    break
                w = line.strip("\n").strip()
                if not w:
                    continue
                w = normalize_text(w, lowercase=lowercase, nfkc=nfkc)
                chars.update(list(w))

        vocab = sorted(chars)
        return cls(vocab=vocab, lowercase=lowercase, nfkc=nfkc)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save tokenizer vocabulary to JSON file.
        
        Args:
            path: Output path for JSON file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "char",
            "lowercase": self.lowercase,
            "nfkc": self.nfkc,
            "vocab": self.itos,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CharTokenizer":
        """
        Load tokenizer from JSON vocabulary file.
        
        Args:
            path: Path to JSON file saved by save().
        
        Returns:
            CharTokenizer instance with loaded vocabulary.
        """
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        
        if payload.get("type") != "char":
            raise ValueError(
                f"CharTokenizer.load expected type=char, got: {payload.get('type')}"
            )
        
        vocab = payload["vocab"]
        lowercase = bool(payload.get("lowercase", True))
        nfkc = bool(payload.get("nfkc", True))

        # Create instance and restore exact vocab ordering
        tok = cls(vocab=vocab, lowercase=lowercase, nfkc=nfkc)
        tok.itos = vocab
        tok.stoi = {t: i for i, t in enumerate(tok.itos)}
        tok._PAD = tok.stoi[tok.PAD_TOKEN]
        tok._BOS = tok.stoi[tok.BOS_TOKEN]
        tok._EOS = tok.stoi[tok.EOS_TOKEN]
        tok._UNK = tok.stoi[tok.UNK_TOKEN]
        tok._vocab_size = len(tok.itos)
        
        return tok

    def encode(self, text: str) -> List[int]:
        """
        Encode text to character IDs.
        
        Does NOT add BOS/EOS automatically.
        """
        text = normalize_text(text, lowercase=self.lowercase, nfkc=self.nfkc)
        return [self.stoi.get(ch, self._UNK) for ch in text]

    def decode(self, ids: List[int]) -> str:
        """
        Decode character IDs to text.
        
        Automatically removes PAD, BOS, EOS tokens.
        """
        out: List[str] = []
        for i in ids:
            if i in (self._PAD, self._BOS, self._EOS):
                continue
            if 0 <= i < len(self.itos):
                out.append(self.itos[i])
            else:
                out.append(self.UNK_TOKEN)
        return "".join(out)
