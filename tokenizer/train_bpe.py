import os

import sentencepiece as spm

vocab_size = 500
HERE = os.path.dirname(os.path.abspath(__file__))

for fk in ["0","1","2","3","4"]:
    corpus = os.path.join(HERE, f"bpe_corpus_fold_{fk}.txt")
    prefix = os.path.join(HERE, f"bpe{vocab_size}_fold_{fk}")
    spm.SentencePieceTrainer.Train(
        input=corpus,
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0, pad_piece="<pad>",
        bos_id=1, bos_piece="<bos>",
        eos_id=2, eos_piece="<eos>",
        unk_id=3, unk_piece="<unk>"
    )
    print(f"Trained tokenizer: {prefix}.model")
