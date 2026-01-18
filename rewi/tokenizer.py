import sentencepiece as spm

import json
import unicodedata
from pathlib import Path


def _norm_text(text: str, *, lowercase: bool = True, nfkc: bool = True) -> str:
    if nfkc:
        text = unicodedata.normalize("NFKC", text)
    if lowercase:
        text = text.lower()
    return text

class BPETokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.PAD = self.sp.pad_id()   # 0
        self.BOS = self.sp.bos_id()   # 1
        self.EOS = self.sp.eos_id()   # 2
        self.UNK = self.sp.unk_id()   # 3
        self.vocab_size = self.sp.vocab_size()

    def encode(self, text: str) -> list[int]:
        # Return ids (no automatic BOS/EOS here, we add them in build_ar_batch)
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        # Remove specials if present
        ids = [t for t in ids if t not in (self.PAD, self.BOS, self.EOS)]
        return self.sp.decode(ids)

    def pieces(self, ids: list[int]) -> list[str]:
        return self.sp.id_to_piece(ids)


class CharTokenizer:
    """Minimal character tokenizer.

    - Encodes a string into character IDs (no automatic BOS/EOS).
    - Decodes IDs back into a string (special tokens removed).
    - Can save/load vocab JSON for reproducible downstream use.
    """

    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(
        self,
        vocab: list[str],
        *,
        lowercase: bool = True,
        nfkc: bool = True,
    ) -> None:
        self.lowercase = lowercase
        self.nfkc = nfkc

        # Ensure specials exist and are first (stable IDs)
        specials = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        seen = set()
        ordered: list[str] = []
        for tok in specials + vocab:
            if tok in seen:
                continue
            seen.add(tok)
            ordered.append(tok)

        self.itos = ordered
        self.stoi = {t: i for i, t in enumerate(self.itos)}

        self.PAD = self.stoi[self.PAD_TOKEN]
        self.BOS = self.stoi[self.BOS_TOKEN]
        self.EOS = self.stoi[self.EOS_TOKEN]
        self.UNK = self.stoi[self.UNK_TOKEN]
        self.vocab_size = len(self.itos)

    @classmethod
    def build_from_text_file(
        cls,
        path: str | Path,
        *,
        lowercase: bool = True,
        nfkc: bool = True,
        max_lines: int | None = None,
    ) -> "CharTokenizer":
        path = Path(path)
        chars: set[str] = set()
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if max_lines is not None and i >= max_lines:
                    break
                w = line.strip("\n").strip()
                if not w:
                    continue
                w = _norm_text(w, lowercase=lowercase, nfkc=nfkc)
                chars.update(list(w))

        vocab = sorted(chars)
        return cls(vocab=vocab, lowercase=lowercase, nfkc=nfkc)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "char",
            "lowercase": self.lowercase,
            "nfkc": self.nfkc,
            "vocab": self.itos,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("type") != "char":
            raise ValueError(f"CharTokenizer.load expected type=char, got: {payload.get('type')}")
        vocab = payload["vocab"]
        # Keep the saved vocab ordering exactly (IDs must match)
        lowercase = bool(payload.get("lowercase", True))
        nfkc = bool(payload.get("nfkc", True))

        tok = cls(vocab=vocab, lowercase=lowercase, nfkc=nfkc)
        # Ensure exact ordering preserved
        tok.itos = vocab
        tok.stoi = {t: i for i, t in enumerate(tok.itos)}
        tok.PAD = tok.stoi[tok.PAD_TOKEN]
        tok.BOS = tok.stoi[tok.BOS_TOKEN]
        tok.EOS = tok.stoi[tok.EOS_TOKEN]
        tok.UNK = tok.stoi[tok.UNK_TOKEN]
        tok.vocab_size = len(tok.itos)
        return tok

    def encode(self, text: str) -> list[int]:
        text = _norm_text(text, lowercase=self.lowercase, nfkc=self.nfkc)
        ids: list[int] = []
        for ch in text:
            ids.append(self.stoi.get(ch, self.UNK))
        return ids

    def decode(self, ids: list[int]) -> str:
        out: list[str] = []
        for i in ids:
            if i in (self.PAD, self.BOS, self.EOS):
                continue
            out.append(self.itos[i] if 0 <= i < len(self.itos) else self.UNK_TOKEN)
        return "".join(out)
