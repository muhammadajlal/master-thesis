import json, os
import sentencepiece as spm

TRAIN_JSON = "/home/woody/iwso/iwso214h/imu-hwr/data/wi_sent_hw6_meta/train.json"
with open(TRAIN_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

ann = data.get("annotations", {})
fold_keys = sorted(ann.keys(), key=lambda x: int(x))  # ["0","1","2","3","4"]

for fk in fold_keys:
    path = f"/home/woody/iwso/iwso214h/imu-hwr/work/REWI_work/tokenizer/bpe_corpus_fold_{fk}.txt"
    count = 0
    with open(path, "w", encoding="utf-8") as out:
        for item in ann[fk]:
            word = item["label"].strip()
            if word:
                out.write(word + "\n")
                count += 1
    print(f"Wrote {count} lines to: {path}")


