# Alignment-Stable Attention for IMU Handwriting Recognition

**Master's Thesis Repository — Muhammad Ajlal Khan (FAU Pattern Recognition Lab)**

Codebase and experimental artifacts for a Master's thesis on **writer-independent (WI) online handwriting recognition (HWR) from IMU time series**. The work builds on the REWI baseline introduced in *"Robust and Efficient Writer-Independent IMU-Based Handwriting Recognition"* (Li et al., iWOAR 2025; arXiv:2502.20954).

The thesis studies six families of decoder strategies for IMU HWR:

- **CTC** — CNN encoder + BiLSTM/Transformer head with CTC loss
- **Autoregressive (AR)** — CNN encoder + Transformer decoder with cross-attention (the canonical baseline carries elementwise gating)
- **Hybrid CTC-AR** — joint training with `λ_AR · L_AR + λ_CTC · L_CTC`
- **Multimodal LM (T5/ByT5)** — CNN encoder + frozen/LoRA-adapted HuggingFace seq2seq head
- **VLM (GPT-2)** — BLIP-2-style: CNN encoder → Q-Former / MLP / Pooling connector → soft prefix → GPT-2 (LoRA)
- **Hybrid CTC + VLM** — auxiliary CTC head over the encoder while the VLM does autoregressive decoding

Additional ablations covered in the codebase: input-corruption regularisation for AR (uniform / bigram / self-confusion / adjacent-swap), scheduled sampling, contrastive alignment loss, parameter-matched architecture comparisons (mini Q-Former, KV multi-view, Conformer encoder, ByT5 decoder), and zero-shot transfer to OnHW equations.

---

## Repository layout

```
work/REWI_work/
├── main.py                     # Train/evaluate one fold
├── evaluate.py                 # 5-fold CV aggregation + MACs/params
├── pretrain_decoder.py         # Decoder-only text-pretraining
│
├── configs/                    # YAML configs grouped by experiment family
│   ├── AR-Baseline*/                       # Vanilla AR transformer baselines
│   ├── AR-InputCorruption*/                # Input-corruption ablation (5 variants)
│   ├── AR-ScheduledSamplingFixed*/         # Fixed-p scheduled sampling
│   ├── hybrid/                             # Hybrid CTC-AR configs
│   ├── H1_hybrid_ctc_vlm/                  # Hybrid CTC + VLM (Phase 4)
│   ├── J2_contrastive_*/                   # CTC + InfoNCE contrastive (Phase 5)
│   ├── K1_ctc_mse, K2_ctc_posterior, ...   # Token alignment / architecture (Phase 6-7)
│   ├── L1_mini_qformer, L2_kv_slim/        # Param-matched VLM connectors
│   ├── M1_byt5_hybrid_mlp/                 # ByT5 byte-level decoder swap
│   ├── N1_conformer_hybrid_mlp/            # Conformer encoder swap
│   └── ...
│
├── rewi/                       # Core library
│   ├── model/                  # Encoders, decoders, dual-head, VLM, Q-Former
│   ├── dataset/                # HRDataset, augmentations, AR/CTC/LM collation
│   ├── training/loops.py       # Training loops per regime (incl. input_corruption + scheduled_sampling)
│   ├── analysis/               # Cross-attention / Grad-CAM / metrics
│   ├── tokenizer/              # Char and SentencePiece BPE
│   └── ...
│
├── analysis/                   # Quantitative + qualitative analysis scripts
├── scripts/                    # Plotting, dictionary builders, repro helpers
├── slurm/                      # Cluster job scripts
└── tokenizer/                  # SentencePiece BPE models
```

---

## Quickstart

### 1) Environment

```bash
conda create -n rewi python=3.12.10
conda activate rewi
pip install -r environment-lock.txt
```

### 2) Working directory

All commands assume `cd work/REWI_work` (or `export PYTHONPATH=/path/to/work/REWI_work`).

### 3) Data layout

MSCOCO-like:

```
data/<dataset>/
├── train.json    # {"annotations": {"<fold>": [{label, filename, id_writer}, ...]}}
├── val.json
└── data/*.csv    # 13-channel semicolon-separated IMU traces
```

The dataset path is set via the YAML key `dir_dataset`; SLURM scripts patch it from `$DATASET`.

### 4) Train one fold

```bash
python main.py -c configs/AR-Baseline/train-ar-baseline-onhw.yaml
```

Outputs land in `<dir_work>/<fold>/<idx_fold>/`:
- `checkpoints/` — `best_cer.pth`, `last.pth`, ...
- `train_<timestamp>.json` — per-epoch metrics
- `train_<timestamp>.log` — loguru log

### 5) Aggregate 5-fold CV

```bash
python evaluate.py -c configs/AR-Baseline/train-ar-baseline-onhw.yaml
```

Globs `**/train_*.json` recursively under `dir_work`, picks per-fold best by validation CER (capped at `epoch − 1` to ignore overshoot from resumes), writes `<dir_work>/results.json` with mean/std for CER and WER plus MACs/params.

### 6) Cluster (SLURM, 5-fold array)

```bash
sbatch --array=0-4 -p rtx3080 --gres=gpu:rtx3080:1 \
       --export=ALL,TRAIN_YAML=AR-Baseline/train-ar-baseline-onhw.yaml,DATASET=onhw_wi_word_rh \
       slurm/train.sbatch
```

The sbatch script patches `idx_fold`, `dir_work`, and `dir_dataset` per-task.

---

## Key configuration knobs

| YAML key | Purpose |
|---|---|
| `arch_en` | Encoder architecture (`blconv_b` is the canonical baseline) |
| `arch_de` | Decoder architecture (`ar_transformer_xs`, `ar_transformer_s`, `t5-small`, `byt5-small`, `vlm`, ...) |
| `idx_fold` | Fold index (0–4, or −1 for all-folds via `train_cv.py`) |
| `dir_dataset` / `dir_work` | Dataset + output paths (patched by sbatch) |
| `epoch` / `epoch_warmup` | 300 / 30 across all current experiments |
| `lr` / `size_batch` | 0.001 / 64 (defaults) |
| `seed` | 42 |
| `dual_head.enabled` | Adds CTC head to AR/LM/VLM models |
| `input_corruption.{mode, p_replace, smoothing, confusion_path}` | AR exposure-bias regularisation |
| `scheduled_sampling.{enabled, p_start, p_end, ramp_epochs}` | Two-pass scheduled sampling |
| `vlm_enabled`, `vlm.{lm_name, num_queries, ...}` | VLM mode |
| `lm_name`, `lm_train_lm`, `lm_unfreeze_epoch`, `lr_enc/proj/lm` | T5/ByT5 mode |

For the retained environment, exact thesis configuration mapping, and artifact provenance, see [ENVIRONMENT.md](ENVIRONMENT.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

---

## Reproducibility

Replication-grade instructions, public-data download (OnHW500), one-command repro script, and offline HuggingFace handling are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

- **OnHW500** data preparation and the public training/evaluation pipeline are reproducible end-to-end. The exact thesis-reported configuration and result families are listed in `REPRODUCIBILITY.md`; other public-task configs are exploratory.
- **Stabilo** datasets (`wi_word_hw6_meta`, `wi_sent_hw6_meta`) are private — results on Stabilo are reported but cannot be redistributed.

---

## Thesis context

Studied dimensions:

- **Decoder type**: CTC vs AR vs Hybrid vs LM-based vs VLM-based
- **Connector capacity / architecture** (VLM): Q-Former / MLP / Pooling / KV multi-view at matched parameter budgets
- **Modality-gap interventions**: sequence-level contrastive alignment (J2/J2p), CTC compress + MSE (K1), CTC posterior reconstruction (K2), and hard-negative encoder regularization (K3)
- **Excluded historical run**: K4 argmax-segment contrastive is retained for provenance but excluded from RQ3 because its unmasked comparison bank does not implement the intended positive-masked InfoNCE or SEA objective; see `REPRODUCIBILITY.md`
- **Architectural swaps**: Conformer encoder (N1), ByT5 byte-level decoder (M1)
- **Exposure-bias remedies (AR)**: scheduled sampling vs input corruption (uniform / bigram-right / bigram-left / self-confusion / adjacent-swap)
- **Codebase-only exploratory transfer:** J-series VLM checkpoints evaluated on unseen OnHW equations; these runs are retained but are not reported in the thesis.

For thesis-ready tables/figures and protocol text:
- [docs/Master-Project-Report.pdf](docs/Master-Project-Report.pdf)
- [docs/Master-Thesis-Proposal.pdf](docs/Master-Thesis-Proposal.pdf)

The aggregated experiment matrix (per-dataset CER/WER, p_replace sweeps, statistical tests) is maintained in `results/EXPERIMENT_TRACKER.md` (outside this directory, in the thesis repo root).

---

## References

- **REWI baseline (foundation):**
  Li, J., Hamann, T., Barth, J., Kämpf, P., Zanca, D., Eskofier, B. (2026). *Robust and Efficient Writer-Independent IMU-Based Handwriting Recognition*. In: Sensor-Based Activity Recognition and Artificial Intelligence (iWOAR 2025), LNCS 16292, Springer. https://doi.org/10.1007/978-3-032-13312-0_16

- **Master's Project report (completed phase, predecessor to thesis):**
  *Improving IMU-Based Online Handwriting Recognition: Upgrading from CTC to Attention-based Autoregressive Decoder.* [docs/Master-Project-Report.pdf](docs/Master-Project-Report.pdf)

- **Thesis proposal / scope:**
  *Alignment-Stable Attention for Writer-Independent IMU Handwriting Recognition via Hybrid CTC–AR Training, Calibrated Decoding, and Pretrained LM Decoder Adaptation.* [docs/Master-Thesis-Proposal.pdf](docs/Master-Thesis-Proposal.pdf)

---

## License

MIT — see [LICENSE.txt](LICENSE.txt).
