import sentencepiece as spm

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
