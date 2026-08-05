# Reproducibility / Replication Guide

This document is the **replication guide** for the thesis experiments: it records data availability, canonical configuration and result locations, statistical analysis, and a public-data sanity-check workflow.

For the project description, repo layout, and general usage of `main.py` / `evaluate.py`, see [README.md](README.md).

The most replication-friendly way to reproduce results is to run the **exact YAML configs** from the repository root and generate `results.json` via `evaluate.py`.

## Data availability

- **STABILO / `wi_word_hw6_meta` and `wi_sent_hw6_meta` (private):** For commercial reasons, these datasets are not published. Their annotations and sensor files must be included in the restricted submission archive made available to the supervisor. The corresponding thesis results are not independently reproducible without that archive.
- **OnHW500 (public):** The public counterpart uses the right-handed writer-independent and writer-dependent subsets of OnHW-words500. Download instructions are provided by Fraunhofer IIS: https://www.iis.fraunhofer.de/de/ff/lv/dataanalytics/anwproj/schreibtrainer/onhw-dataset.html

We use a MSCOCO-like structure for training and evaluation. After downloading OnHW, convert it with `scripts/onhw.ipynb`; set `dir_raw`, `dir_out`, and `writer_indep`/`writer_dep` in the notebook before executing all cells.

### Data Layout

After running the notebook, the MSCOCO-like dataset structure is available under `<dir_out>/`:
```
data/
└── onhw_wd_word_rh/
    ├── train.json          # Training annotations (by fold)
    ├── val.json            # Validation annotations (by fold)
    └── data/               # Sensor CSV files
```

## What “replication” means here

- Training is performed per-fold (cross validation), and results are aggregated.
- Model selection is **per-fold best validation CER**, and final numbers are reported as mean ± std across folds.
- Aggregation + MACs/params export is handled by `evaluate.py`, writing `<dir_work>/results.json`.

## Environment

Run commands from the repository root (the folder created by `git clone`).
The retained experiment environment is documented in `ENVIRONMENT.md` and
pinned in `environment-lock.txt`. Install the pinned Python packages with:

```bash
python -m pip install -r environment-lock.txt
```

The key retained versions are Python 3.12.10, PyTorch 2.9.1+cu128,
Transformers 4.57.3, PEFT 0.18.1, JiWER 4.0.0, NumPy 2.2.5, SciPy 1.17.0,
and timm 1.0.24. The local GPT-2 and ByT5-small files are identified by
`model-assets.sha256`; those file hashes are authoritative
when an upstream model revision is unavailable.

The configurations set `seed: 42`, but GPU training can remain nondeterministic
depending on CUDA/cuDNN kernels. Representative canonical SLURM logs identify
Tesla V100-PCIE-32GB GPUs; job-specific logs remain authoritative. Exact
wall-clock time can differ and is not used as thesis evidence.

## Thesis inference MAC artifact

The Chapter 6 HWR-GPT MAC rows are generated from the repository root with:

```bash
python analysis/scripts/vlm_macs_for_thesis.py
```

The script profiles the OnHW-words500 reference shape (`T=1024`) and the
rounded-up mean GPT-2 output length of three tokens. It follows true greedy
Hugging Face generation with key/value caching and fixes both the minimum and
maximum new-token counts to three so early EOS output cannot change the traced
shape. The training-only auxiliary CTC head is excluded unless the CTC
posterior is an inference-time connector input.

The script writes its generated output to the workspace-level `../../results/thesis_vlm_macs.json`. A byte-identical handover mirror is retained inside this repository at `results/thesis_vlm_macs.json`; both copies have SHA-256 `8eb81196eed60335ff7850776047f216322d8f7560b71198d8aaca285ae38ed5`. The workspace-level file is the canonical generation output, and the repository copy is the immutable handover artifact. The JSON was regenerated twice with an identical checksum. These are operation
counts, not latency, peak-memory, or variable-length deployment measurements.

**Row naming.** The artifact names `K5_kv_multiview` as `Legacy K5 KV Multi-View`;
that four-view, 8.08 M-trainable control is not reported in the thesis. The thesis's
two-view, 5.12 M-trainable **Gated Multi-View** connector is `L2_kv_slim` and appears
as `Gated Multi-View (thesis L2)` (4.874 B MACs). The explicit labels prevent the
two configurations from being associated by their rounded 4.9 B MAC values.

## Partial one-command public replication

We provide a helper script that:
1) patches dataset/work paths into a config,
2) runs sequential cross-validation training,
3) runs `evaluate.py` to write `results.json`.

From the repository root:

```bash
bash scripts/repro/reproduce_tables.sh \
  --data-root data \
  --out-root results/repro
```

This helper sanity-checks the public data and training pipeline by reproducing **OnHW500 WI** and **OnHW500 WD** experiments for:

- CNN–ARTransformer (Decoder not-pretrained)
- CNN–t5-small (decoder-only)


It does not reproduce every thesis table. The exact Chapter 6 configuration and result mapping is recorded below.

## Getting `t5-small` weights (offline-first)

Our code defaults to offline loading for HuggingFace models (`lm_local_files_only: true`).

To download `t5-small` once (when internet is available) and store it in the repo under `assets/hf_models/`:

```bash
bash scripts/repro/download_t5_small.sh
```

This downloads into:
- `assets/hf_models/t5-small`

The corresponding YAML fields are:

```yaml
lm_name: assets/hf_models/t5-small
lm_local_files_only: true
```

If you prefer online loading via the HuggingFace API, set:

```yaml
lm_name: t5-small
lm_local_files_only: false
```

Model page (reference): https://huggingface.co/t5-small

### Legacy pretrained AR decoder helper

The helper's legacy “pretrained decoder” condition additionally requires a decoder-only pretraining checkpoint (`best_loss.pth`). This is separate from the GPT-2 experiments in Chapter 6.
The checkpoint depends on an external text corpus; the following procedure can be applied to word- or sentence-level pretraining by selecting the corresponding inputs.

### Download the source sentence/word lists (News 2024, 1M)
The input files come from the Leipzig Corpora Collection / Wortschatz Leipzig downloads:
https://wortschatz-leipzig.de/de/download/

We use the **News** corpora for year **2024** at size **1M** (measured in number of sentences on the website). Download both:
- English: https://wortschatz-leipzig.de/de/download/eng  → section **News** → row **2024** → column **Download 1M**
- German: https://wortschatz-leipzig.de/de/download/deu → section **News** → row **2024** → column **Download 1M**

After downloading, extract the archive (format depends on the download; typically a `.tar.gz`):

```bash
tar -xzf eng_news_2024_1M.tar.gz
tar -xzf deu_news_2024_1M.tar.gz
```

Then locate the `*-sentences.txt` file inside each extracted folder and place/rename it to match the expected inputs below:
- `eng_news_2024_1M-sentences.txt`
- `deu_news_2024_1M-sentences.txt`

Note: These downloads are subject to the provider's terms of use (see the download page).

## Inputs
- `eng_news_2024_1M-sentences.txt`
- `deu_news_2024_1M-sentences.txt`

These files are sentence-per-line, optionally prefixed with a numeric rank:

```
1  $0.07 of every dollar for public officers' pensions.
2  $100 million boost for new pro league.
...
```

Note: Some lines have a leading `$` character which is stripped during processing.

## What was filtered
1. Read all unique sentence labels from:
   - `../../../../data/wi_sent_hw6_meta/train.json`
   - `../../../../data/wi_sent_hw6_meta/val.json`

   In this dataset, the label is stored in the annotation field `label`.

2. Normalize everything the same way (by default):
   - Unicode NFKC normalization
   - lowercase
   - collapse whitespace runs

3. Build a mixed set of sentences from EN∪DE, then remove anything that matches a dataset label.

## Outputs
- `mixed_en_de_no_wi_sent_hw6_meta.txt`
  - Final mixed dictionary (**1,999,971 sentences**, one per line)
  - **Already has wi_sent_hw6_meta label-sentences removed**

- `removed_due_to_leakage.txt`
  - Sentences that were removed because they matched a dataset label
  - **1 sentence** removed: `"three people were injured."`
  - Sorted for easy inspection

## How to reproduce
From the repository root:

```bash
python3 scripts/tools/build_mixed_dictionary.py \
  --kind sent \
  --en assets/dictionaries/sent/eng_news_2024_1M-sentences.txt \
  --de assets/dictionaries/sent/deu_news_2024_1M-sentences.txt \
  --dataset data/wi_sent_hw6_meta \
  --out assets/dictionaries/sent/mixed_en_de_no_wi_sent_hw6_meta.txt \
  --out-removed assets/dictionaries/sent/removed_due_to_leakage.txt
```

The script used is: `scripts/tools/build_mixed_dictionary.py`.


If you have a compatible `best_loss.pth`, you can include the pretrained-decoder runs:

```bash
bash scripts/repro/reproduce_tables.sh \
  --data-root data \
  --out-root results/repro \
  --pretrained-decoder-ckpt /abs/path/to/best_loss.pth
```

## Thesis experiment manifest

### Thesis artifact provenance

This table is the compact entry point from reader-facing evidence to the script and retained inputs that generate it. The detailed configuration-to-result mappings follow below. Paths are relative to this repository unless prefixed by `$RESULTS_ROOT`.

| Thesis artifact family | Generator or verifier | Retained inputs / result family |
|---|---|---|
| Chapter 5 migration, gating, capacity, and scheduled-sampling tables | `evaluate.py` | Chapter 5 canonical `results.json` families listed below; scheduled sampling uses `configs/_ss_xs_frozen/` |
| Noise-rate and noise-mode figures | `analysis/scripts/plot_corruption_p_sweep_dual.py`; `analysis/scripts/plot_corruption_modes_bars_dual.py` | `Baseline-AR-XS-InputCorruption-*` and matched HWRFormer-L result families |
| Classical auxiliary-CTC weight-selection figure | `analysis/scripts/plot_lambda_sweep_dual.py` | `train_element_word_hybrid_{01..10}_xs_*` and HWRFormer-L `Baseline-Hybrid/` sweep results |
| Paired noise table, ECDF figure, and Appendix E decomposition tables | `analysis/scripts/cascade_analysis.py` | `analysis/quant_all_val_predictions_ar_vs_noise_xs.csv` and its five-fold paired prediction exports |
| Prefix-perturbation figure | `analysis/scripts/plot_prefix_perturbation.py` | Per-fold `eval_tf_perturbation_p{000,005,010,015,020}.json` files under the HWRFormer and noise-trained result families |
| Single-corruption recovery figure | `analysis/scripts/plot_single_corruption_recovery.py` | Per-fold `eval_single_corruption.json` files under the same two result families |
| Cosine, PCA, and attention diagnostics | `analysis/scripts/compare_ar_hybrid.py`; `analysis/scripts/render_cosine_grid_offline.py` | Retained Fold-0 checkpoints, prediction CSVs, and versioned sample-selection metadata |
| RQ2 calibration figures | `scripts/decode_study_thesis_analysis_xs_noleak.py` | Forty retained per-fold `metrics.json` files in the two `decode_study_xs_full_*_noleak` families |
| RQ2 decoding tables (WI cross-model at matched N-best = 4; cross-dataset transfer) | `analysis/scripts/summarize_decode_crossds.py` (gated aggregation; emits the exact LaTeX rows) | Per-fold `metrics.json`/`config.json`/`predictions.json` in `decode_study_xs_full_ar{,_wd,_privword,_privsent}`, `decode_study_xs_full_{noise,hybrid}`, `decode_study_xs_full_{ar,hybrid}_noleak_a0`, `decode_rescore_n4_{wi,wd,privword,privsent}`, and `decode_rescore_n4_wi_{noise,hybrid}` |
| Appendix decoding-runtime table (Tab. C.5) | `analysis/scripts/summarize_decode_runtimes.py` (gated: config asserts, CER agreement <= 0.1 pp vs canonical cells, uniform batch; emits the exact LaTeX rows) | Per-fold `metrics.json` in `decode_timing_v100_{wi,wd,privword,privsent}` produced by `slurm/decode_timing_v100.sbatch` (single-hardware v100 re-decode of every Tab. 5.13 cell; only runtime fields are reported) |
| Chapter 6 means, paired effects, intervals, and fold-variation summary | `scripts/chapter6_analysis.py` | Chapter 6 `results.json` families in the manifest below; deterministic JSON: `results/thesis_chapter6_analysis.json` |
| HWR-GPT auxiliary-CTC sweep figures | `analysis/scripts/plot_H1_lambda_sweep.py`; `analysis/scripts/plot_H1_lambda_sweep_word.py` | H1 lambda-sweep results for OnHW and private words |
| Sequence-alignment UMAP figures | `scripts/plot_contrastive_comparison_thesis.py`; `scripts/plot_v3c_contrastive_comparison_thesis.py` | Corrected J2 Fold-0 embedding dumps under `analysis/embedding_viz/` |
| Hybrid-noise aggregate artifact | `analysis/scripts/hybrid_noise_summary.py` | Five noise modes, four datasets, and five folds; output `results/thesis_hybrid_noise_l01.json` |
| Classical and HWR-GPT MAC rows | `analysis/scripts/mac_token_budget.py`; `analysis/scripts/vlm_macs_for_thesis.py` | `results/thesis_mac_token_budget.json` and the SHA-pinned HWR-GPT MAC artifact described above |

All paths below are relative to the repository root. In result patterns, `DATA` denotes either `onhw_wi_word_rh` or `wi_word_hw6_meta`. A directory ending in `_noctc` denotes auxiliary CTC weight λ = 0, `_lam02` denotes λ = 0.2, and no suffix denotes λ = 0.6, unless the row states otherwise. Each terminal result directory contains `results.json`; the per-fold subdirectories preserve the generated training YAML and selected-checkpoint metadata.

Set the archived result root as:

```bash
RESULTS_ROOT=../../results/hwr2
```

### Chapter 6 configurations and results

| Thesis condition | Canonical configuration | Result family below `$RESULTS_ROOT` |
|---|---|---|
| MLP, pretrained GPT-2 | `configs/vlm_ablation/train-vlm-A1-mlp-pretrained.yaml` | `Ablations-MMLM/GPT-2/AR-only/vlm_ablation_A1_mlp_pretrained/vlm__DATA` |
| MLP, random GPT-2 | `configs/vlm_ablation/train-vlm-A2-mlp-random.yaml` | `Ablations-MMLM/GPT-2/AR-only/vlm_ablation_A2_mlp_random/vlm__DATA` |
| Linear, pretrained GPT-2 | `configs/vlm_ablation/train-vlm-A3-linear-pretrained.yaml` | `Ablations-MMLM/GPT-2/AR-only/vlm_ablation_A3_linear_pretrained/vlm__DATA` |
| Pool-MLP, pretrained GPT-2 | `configs/vlm_ablation/train-vlm-B1-pooling-pretrained.yaml` | `Ablations-MMLM/GPT-2/AR-only/vlm_ablation_B1_pooling_pretrained/vlm__DATA` |
| Conv-Pool, pretrained GPT-2 | `configs/G1_minimal_connector/train-G1b-conv-{onhw,word}.yaml` | `Ablations-MMLM/GPT-2/AR-only/G1b_conv_pool/vlm__DATA` |
| Frozen mini-Q-Former encoder diagnostics | `configs/_f1f2_ablation/train-F{1,2}-qenc-{onhw,word}.yaml` | `Ablations-MMLM/GPT-2/AR-only/{F1_frozen_enc_mlp,F2_vlm_enc_ar}/*_qenc_ablation` |
| Frozen HWRFormer encoder diagnostics | `configs/_f1f2_ablation/train-F{1,2}-hwenc-{onhw,word}.yaml` | `Ablations-MMLM/GPT-2/AR-only/{F1_frozen_enc_mlp,F2_vlm_enc_ar}/*_hwenc_ablation` |
| MLP + auxiliary CTC | `configs/H1_hybrid_ctc_vlm/train-H1-mlp-{onhw,word}.yaml`; OnHW λ = 0.2: `configs/H1_hybrid_ctc_vlm/lambda_sweep/train-H1-mlp-onhw-lam02.yaml`; private λ = 0.2: `configs/_lam02_ch6/H1_hybrid_ctc_vlm/train-H1-mlp-word_lam02.yaml` | `Ablations-MMLM/GPT-2/Hybrid/H1_hybrid_mlp/vlm__DATA{,_lam02}`; the canonical OnHW λ = 0.2 result is archived under `H1_LambdaSweep/H1_hybrid_mlp_lam02__onhw_wi_word_rh` |
| Pool-MLP + auxiliary CTC | `configs/H1_hybrid_ctc_vlm_pooling/train-H1-pooling-{onhw,word}.yaml`; corresponding `_lam02_ch6` configs | `Ablations-MMLM/GPT-2/Hybrid/H1_hybrid_pooling/vlm__DATA{,_lam02}` |
| Sequence-level contrastive MLP/Pool | λ = 0.6: `configs/J2_contrastive_{mlp,pooling}/`; λ = 0.2: `configs/_lam02_ch6/J1_contrastive_{mlp,pooling}/` | λ = 0.6: `Ablations-MMLM/GPT-2/Hybrid-Contrastive/J2_contrastive_{mlp,pooling}/vlm__DATA`; λ = 0.2: `.../J1_contrastive_{mlp,pooling}/vlm__DATA_lam02`. **Do not use** `J1_contrastive_*/vlm__DATA` (no suffix): those λ = 0.6 base runs are superseded — see "Superseded contrastive runs" below |
| Argmax compression + MSE | `configs/K1_ctc_mse/`; corresponding `_lam02_ch6` configs | `Ablations-MMLM/GPT-2/Hybrid/K1_ctc_mse/vlm__DATA{,_lam02}` |
| Posterior reconstruction | `configs/K2_ctc_posterior/`; corresponding `_lam02_ch6` configs | `Ablations-MMLM/GPT-2/Hybrid/K2_ctc_posterior/vlm__DATA{,_lam02}` |
| **K4 argmax-segment run -- EXCLUDED FROM RQ3** | `configs/K4_sea_contrastive/`; corresponding `_lam02_ch6` configs | Historical results retained under `Ablations-MMLM/GPT-2/Hybrid-Contrastive/K4_sea_contrastive/`; see the exclusion record below |
| Lightweight Q-Former | `configs/L1_mini_qformer/`, `configs/_noctc_ch6/L1_mini_qformer/`, and corresponding `_lam02_ch6` configs | `Ablations-MMLM/GPT-2/Hybrid/L1_mini_qformer/vlm__DATA{,_noctc,_lam02}` |
| Gated query connector | `configs/L2_kv_slim/`, `configs/_noctc_ch6/L2_kv_slim/`, and corresponding `_lam02_ch6` configs | `Ablations-MMLM/GPT-2/Hybrid/L2_kv_slim/vlm__DATA{,_noctc,_lam02}` |
| ByT5 control | `configs/M1_byt5_hybrid_mlp/train-M1-byt5-{onhw,word}.yaml` | `Ablations-MMLM/byt5-small/Hybrid/M1_byt5_hybrid_mlp/vlm__DATA` |
| Shallow-Conformer control | `configs/N1_conformer_hybrid_mlp/train-N1-conformer-{onhw,word}.yaml` | `Ablations-MMLM/GPT-2/Hybrid/N1_conformer_hybrid_mlp/vlm__DATA` |

Brace notation in the table is shorthand for the listed literal alternatives; it is not a shell command. The result paths used by the paired comparisons in `scripts/chapter6_analysis.py` are encoded in that script. The alignment and auxiliary-CTC sensitivity paths are specified explicitly in the manifest above.

### Excluded K4 argmax-segment run

**Status: EXCLUDED FROM RQ3.** Source audit of `rewi/training/auxiliary_losses.py` showed that the separately prepended designated target also occurs in the comparison bank and that all same-token occurrences remain in the denominator. For normalized anchor `u`, token similarities `s_t`, bank occurrence counts `c_t`, target token `y`, and logit scale `alpha`, the as-run objective is

`-alpha*s_y + log((1+c_y)*exp(alpha*s_y) + sum_{t!=y} c_t*exp(alpha*s_t))`.

The historical run is mathematically defined, but it does not test positive-masked per-position InfoNCE or SEA and is not used in the thesis RQ3 answer.

- Code: `rewi/training/auxiliary_losses.py:322-447`, `rewi/model/vlm_model.py:532-539`, `rewi/dataset/lm_collate.py:87-116`
- Configs: `configs/K4_sea_contrastive/`, `configs/_lam02_ch6/K4_sea_contrastive/`
- OnHW selected result: `Ablations-MMLM/GPT-2/Hybrid-Contrastive/K4_sea_contrastive/vlm__onhw_wi_word_rh_lam02/results.json`
- Private-word selected result: `Ablations-MMLM/GPT-2/Hybrid-Contrastive/K4_sea_contrastive/vlm__wi_word_hw6_meta/results.json`

The result files and configurations are retained for provenance and must not be silently relabeled, overwritten, or included in active Chapter 6 comparisons.

### Model naming: thesis name to `arch_de`

The thesis model names do not match the historical directory names. **`arch_de` is the
authoritative discriminator; directory names are not.** Read this table before using any
Chapter 5 path:

| Thesis name | `arch_de` | Decoder | Params (OnHW word) | Historical marker |
|---|---|---|---|---|
| REWI | `bilstm_wide` | 3-layer BiLSTM + CTC head | 4.64 M | `Baseline-REWI/` |
| **HWRFormer** | **`ar_transformer_xs`** | L = 2, d_ff = 896, d = 256, H = 4 | 4.64 M (elementwise) | `XS` in the path |
| **HWRFormer-L** | **`ar_transformer_s`** | L = 4, d_ff = 1024, d = 256, H = 4 | 6.94 M (elementwise) | no `XS` in the path |

The `XS` suffix therefore marks the thesis's **primary** model (HWRFormer), not a smaller
control; paths without `XS` are the **larger** capacity control (HWRFormer-L). All reported
Chapter 5 models use `arch_en: blconv_b`. Configurations naming `blconv_l`
(`Baseline-AR-blconv_l`) or `transformer_{s,xs}` CTC decoders are exploratory work that no
thesis table reports.

### Chapter 5 conditions and results

Every row below was verified by recomputing the thesis table values from the archived
`results.json`. Where a configuration directory mixes architectures, the file glob is given.

| Thesis condition | `arch_de` | Canonical configuration | Result family below `$RESULTS_ROOT` |
|---|---|---|---|
| REWI reference | `bilstm_wide` | `configs/Baseline-REWI/` | `Baseline-REWI/bilstm_wide__DATA` |
| HWRFormer, elementwise gate (locked reference) | `ar_transformer_xs` | `configs/AR-Baseline/train-ar-baseline-xs-*.yaml` | `Baseline-AR-XS-blconv_b/ar_transformer_xs__DATA` |
| HWRFormer, ungated | `ar_transformer_xs` | `configs/AR-Baseline/train-ar-xs-ungated-*.yaml` | `Baseline-AR-XS-Ungated/ar_transformer_xs__DATA` |
| HWRFormer, headwise gate | `ar_transformer_xs` | `configs/AR-Baseline/train-ar-xs-headwise-*.yaml` | `Baseline-AR-XS-HeadwiseGating/ar_transformer_xs__DATA` |
| HWRFormer-L, elementwise gate | `ar_transformer_s` | `configs/AR-Baseline-WD/`, `configs/AR-Baseline-Equations/` and the matching OnHW-WI/private variants | `Baseline-AR-ElementwiseGating*/ar_transformer_s__DATA` |
| HWRFormer-L, ungated | `ar_transformer_s` | `configs/AR-Baseline/train-ar-ungated-*.yaml` | `Baseline-AR-Ungated/ar_transformer_s__DATA` |
| HWRFormer-L, headwise gate | `ar_transformer_s` | `configs/AR-Baseline/train-ar-headwise-*.yaml` | `Baseline-AR-HeadwiseGating/ar_transformer_s__DATA` |
| HWRFormer noise modes at p = 0.15 | `ar_transformer_xs` | `configs/AR-InputCorruption-XS/` | `Baseline-AR-XS-InputCorruption-{uniform,bigramright,bigramleft,selfconf,adjacentswap}/` |
| HWRFormer noise-rate sweep | `ar_transformer_xs` | `configs/AR-InputCorruption-Sweep-XS/` | `Baseline-AR-XS-InputCorruption-Sweep-blconv_b/…__p0pXX` |
| HWRFormer-L noise | `ar_transformer_s` | `configs/AR-InputCorruption/` | `Baseline-AR-InputCorruption-blconv_b/`, `…-WD-uniform/` |
| Hybrid HWRFormer, λ = 0.1 | `ar_transformer_xs` | `configs/hybrid-xs/` | `train_element_word_hybrid_01_xs_{onhw_wi,onhw_wd,stabilo,stabilo_sent}/` |
| Hybrid HWRFormer-L λ sweep (λ = 0.6 operating point) | `ar_transformer_s` | `configs/hybrid/` | `Baseline-Hybrid/train_element_word_hybrid_{01..10}/`, `train_element_word_hybrid_06_*/` |
| Hybrid + noise, λ = 0.1 | `ar_transformer_xs` | `configs/HybridInputCorruption-XS-L01/` | `HybridInputCorruption-XS-L01_{uniform,bigram_left,bigram_right,self_confusion,adjacent_swap}/` |
| Scheduled-sampling probe | `ar_transformer_xs` | `configs/_ss_xs_frozen/train-ar-{noTF,ssfixed}-{onhw-word,stabilo-word,stabilo-sent}.yaml` | `Baseline-AR-XS-{NoTeacherForcing,ScheduledSamplingFixed}-blconv_b/ar_transformer_xs__{onhw_wi_word_rh,wi_word_hw6_meta,wi_sent_hw6_meta}` |
| Inference-time decoding study | `ar_transformer_xs` | `configs/decode_study/` | `decode_study_xs_full_{ar,hybrid}_noleak*/stage*__fold{0..4}/metrics.json` |

> **Scheduled-sampling note.** Table 5.4 uses the six immutable HWRFormer configurations in
> `configs/_ss_xs_frozen/`: the immediate ramp `p: 0 -> 1` and fixed `p = 0.15` schedules on
> OnHW WI, private words, and private sentences. Each corresponding `ar_transformer_xs__*`
> result directory contains five generated fold configurations and a canonical `results.json`.
> The delayed-ramp and delayed-abrupt configurations in the same directory are separate curriculum
> controls and are intentionally excluded from the thesis. The older `ar_transformer_s` scheduled-
> sampling configurations and result families are retained as historical HWRFormer-L artifacts but
> are not reported in the thesis table.

Preserve these YAMLs together with the generated fold configurations and `results.json`;
directory names alone are not a substitute for the archived parameters.

### Per-dataset leak-free KenLM models (privacy-critical)

The decoding study uses one character 5-gram KenLM per dataset and validation fold, stored as `<lmdir>/fold_<k>/{corpus.txt, char_5gram.arpa, char_5gram.binary}`:

| Directory | Dataset |
|---|---|
| `lm_noleak/` | `onhw_wi_word_rh` |
| `lm_noleak_wd/` | `onhw_wd_word_rh` |
| `lm_noleak_privword/` | `wi_word_hw6_meta` |
| `lm_noleak_privsent/` | `wi_sent_hw6_meta` |

All four directories are **deliberately untracked** (`.gitignore` rule `lm_noleak*/`): the private-dataset variants contain raw private label text in `corpus.txt` and in the ARPA vocabulary, and must never reach a public remote. They are derived artifacts; regenerate deterministically with the vendored KenLM (`vendor/kenlm`):

```bash
python scripts/train_char_lm.py --dataset ../../data/<dataset> --per_fold --order 5 --outdir <lmdir>
```

The `--per_fold` corpus is built leak-free from the `val.json` partitions of all folds except the test fold (using `train.json[k]` keys would duplicate samples and inflate word frequencies; see the note in `scripts/train_char_lm.py`).

Rescoring provenance: the decoding-study selection grid used `N_best in {8, 16, 32}`, but N-best generation sets the search beam, so the `stageC1_*_N8_*` runs decode at beam 8 rather than the calibrated beam 4. The reported operating point is therefore the matched re-run at `N_best = B_beam = 4` (`decode_rescore_n4_*` and `decode_rescore_n4_wi_{noise,hybrid}` families, produced by `slurm/decode_rescore_n4.sbatch` and `slurm/decode_rescore_n4_models_wi.sbatch`). The N=8 runs are retained as the documented grid point and are superseded for reporting because they combined wider search and list generation with reranking.

### Retained experiment families not reported in the thesis

The repository contains exploratory and historical families beyond the final evidence chain. They are retained for provenance but must not be used to reconstruct a thesis table unless a manifest row above names them explicitly:

- `decode_crossds_{hybrid,noise}_{wd,privword,privsent}` are complete five-fold decoding runs of the hybrid and noise-trained recognizers on the three transfer datasets (defense backup material). The thesis restricts the cross-dataset transfer analysis to plain HWRFormer; these families are measured but unreported, not untested.

- `configs/HybridInputCorruption-XS/` and matching `HybridInputCorruption-XS_*` results use hybrid settings other than the reported HWRFormer lambda = 0.1 matrix.
- `configs/HybridInputCorruption/` and matching unsuffixed result families are HWRFormer-L hybrid-noise explorations and are not reported.
- Configuration or result families containing `Equations` target the OnHW equations tasks and are not part of the thesis evaluation.
- `configs/zero_shot/` and matching ZeroShot result families are exploratory transfer work and are not reported.
- `configs/AR-SS-DelayedRamp/`, `configs/AR-SS-DelayedAbrupt/`, and delayed scheduled-sampling results are curriculum controls excluded from the final scheduled-sampling table.
- Legacy `configs/AR-NoTeacherForcing/` and `configs/AR-ScheduledSamplingFixed/` use HWRFormer-L (`ar_transformer_s`). The thesis table instead uses the immutable HWRFormer (`ar_transformer_xs`) snapshot in `configs/_ss_xs_frozen/`.

### Superseded contrastive runs

`Ablations-MMLM/GPT-2/Hybrid-Contrastive/J1_contrastive_{mlp,pooling}/vlm__{onhw_wi_word_rh,wi_word_hw6_meta}`
(the λ = 0.6 base runs, trained 2026-03-22/23) are **superseded and must not be reported**. They
predate the contrastive logit-scale fix: under commit `dc8c03b`,
`InBatchContrastiveLoss` initialized `log_tau = log(0.07)`, giving a logit scale of τ = 0.07
instead of the intended τ = 1/0.07 = 14.29. That scale flattens the InfoNCE softmax and weakens
the objective. On private words, the pre-fix MLP/Pool-MLP cosine similarities are 0.03/0.06,
matching their respective no-alignment baselines at the reported precision, versus 0.27/0.28
for the corrected runs. The corrected implementation matches the thesis methodology, which
specifies a learnable logit scale initialized to log(1/0.07).

The runs the thesis reports both use the corrected implementation:

- λ = 0.6 → `J2_contrastive_{mlp,pooling}/` (trained 2026-03-24/25 with the corrected
  working-tree implementation, subsequently committed as `78abeb3` on 2026-03-27).
- λ = 0.2 → `J1_contrastive_{mlp,pooling}/vlm__DATA_lam02` (trained 2026-06-28, long after the fix;
  the `J1_` prefix reflects only the configuration lineage cloned by
  `scripts/clone_ch6_configs_lam02.py`, not the March code).

The Fold-0 embedding geometry (`tab:contrastive-alignment`) and both UMAP figures are likewise
generated from the `J2_*` dumps under `analysis/embedding_viz/`.

## Statistical reproduction

The deterministic Chapter 6 analysis validates five folds per central condition and reproduces means, paired fold differences, fold signs, and two-sided 95% Student t intervals:

```bash
python scripts/chapter6_analysis.py --results-root ../../results/hwr2
python scripts/chapter6_analysis.py --results-root ../../results/hwr2 --json
```

Run the command twice and compare the output before submission. The intervals are exploratory because operating-point selection and reporting use the same validation folds.

## Producing the thesis result artifacts

- Training (per fold): `python main.py -c <config.yaml>`
- Sequential CV wrapper: `python scripts/others/train_cv.py -c <config.yaml> -m main.py` (requires `idx_fold: -1`)
- Aggregation and MAC/parameter export: `python evaluate.py -c <config.yaml>`; this writes `<dir_work>/results.json`

For an auditable submission, record the code revision with `git rev-parse HEAD` and preserve any intentional uncommitted patch. The restricted handover must include the private inputs, exact YAMLs, generated per-fold configurations, selected-checkpoint metadata, all canonical `results.json` files, the analysis scripts, and package/environment information. Public-data replication and private-data verification should produce the same table-level means up to documented hardware non-determinism.
