# Configuration index

Every experiment is a YAML file consumed by `main.py` (one fold) or `train_cv.py`
(all folds). Paths inside the YAMLs use three placeholders that are expanded at
load time (`rewi/utils.py: load_cfg`):

| Placeholder | Default | Meaning |
|---|---|---|
| `${REPO}` | repository root | code, `assets/hf_models/`, tokenizer |
| `${DATA_ROOT}` | `${REPO}/data` | datasets in the MSCOCO-like layout |
| `${RESULTS_ROOT}` | `${REPO}/results` | run outputs (`hwr2/<group>/<arch>__<dataset>/…`) |

Set `DATA_ROOT` / `RESULTS_ROOT` in the environment to use external locations.

## Start here: `examples/`

| Config | Model | Thesis table |
|---|---|---|
| `examples/rewi_ctc_onhw_wi.yaml` | REWI reference (CNN + BiLSTM + CTC) | Tab. 5.1 |
| `examples/hwrformer_onhw_wi.yaml` | HWRFormer (CNN + AR Transformer, elementwise SDPA gate) | Tab. 5.1 |
| `examples/hwrformer_noise_injection_onhw_wi.yaml` | HWRFormer + noise injection (uniform, p = 0.15) | Tab. 5.2 |
| `examples/hybrid_hwrformer_onhw_wi.yaml` | Hybrid HWRFormer (auxiliary CTC, λ = 0.1) | Tab. 5.6 |
| `examples/hwr_gpt_hybrid_mlp_onhw_wi.yaml` | HWR-GPT (MLP connector + LoRA GPT-2 + auxiliary CTC) | Tab. 6.4 |

Swap `onhw_wi_word_rh` for `onhw_wd_word_rh` in `dir_dataset` and `dir_work` for the
writer-dependent split. The private STABILO configs (`*stabilo*`, `*word*`, `*sent*`)
are kept for provenance but need data that is not released.

## Thesis experiment families (Chapter 5, classical)

| Thesis condition | Config directory | Result family under `${RESULTS_ROOT}/hwr2/` |
|---|---|---|
| REWI reference | `examples/rewi_ctc_onhw_wi.yaml`, `Baseline-REWI/` | `Baseline-REWI/` |
| HWRFormer (elementwise / ungated / headwise) | `AR-Baseline/train-ar-baseline-xs-*`, `train-ar-xs-ungated-*`, `train-ar-xs-headwise-*` | `Baseline-AR-XS-blconv_b/`, `Baseline-AR-XS-Ungated/`, `Baseline-AR-XS-HeadwiseGating/` |
| HWRFormer-L capacity control | `AR-Baseline/train-ar-baseline-*`, `AR-Baseline-WD/` | `Baseline-AR-ElementwiseGating*/` |
| Parameter-matched CNN-Transformer-CTC | `AR-Baseline/train-transformer-xs-ctc-*-matched-masked.yaml` | `Baseline-Transformer-XS-CTC-Matched-Masked/` |
| Noise-injection modes (p = 0.15) | `AR-InputCorruption-XS/` | `Baseline-AR-XS-InputCorruption-{uniform,bigramright,bigramleft,selfconf,adjacentswap}/` |
| Noise-rate sweep | `AR-InputCorruption-Sweep-XS/` (HWRFormer-L: `AR-InputCorruption-Sweep/`) | `Baseline-AR-XS-InputCorruption-Sweep-blconv_b/…__p0pXX` |
| Hybrid λ sweep + retained λ = 0.1 | `hybrid-xs/` (HWRFormer-L: `hybrid/`) | `train_element_word_hybrid_NN_xs_*/`, `Baseline-Hybrid/` |
| Hybrid + noise (λ = 0.1) | `HybridInputCorruption-XS-L01/` | `HybridInputCorruption-XS-L01_*/` |
| Scheduled-sampling controls | `_ss_xs_frozen/` (immutable snapshot) | `Baseline-AR-XS-{NoTeacherForcing,ScheduledSamplingFixed}-blconv_b/` |
| Inference-time decoding study (beam, KenLM, neural LM, rescoring) | `decode_study/` + `scripts/run_decode_*.sh` | `decode_study_xs_full_*`, `decode_rescore_n4_*` |

## Thesis experiment families (Chapter 6, HWR-GPT)

| Thesis condition | Config directory | Result family |
|---|---|---|
| Connector baselines, pretrained vs random GPT-2 | `vlm_ablation/`, `G1_minimal_connector/` | `Ablations-MMLM/GPT-2/AR-only/…` |
| Frozen-encoder diagnostic | `_f1f2_ablation/` | `…/AR-only/{F1_frozen_enc_mlp,F2_vlm_enc_ar}/` |
| Auxiliary CTC (MLP, Pool-MLP), λ sweeps | `H1_hybrid_ctc_vlm/`, `H1_hybrid_ctc_vlm_pooling/`, `_lam02_ch6/` | `…/Hybrid/H1_hybrid_{mlp,pooling}/`, `H1_LambdaSweep/` |
| Lightweight Q-Former, Gated Multi-View | `L1_mini_qformer/`, `L2_kv_slim/`, `_noctc_ch6/` | `…/Hybrid/{L1_mini_qformer,L2_kv_slim}/` |
| Sequence-level contrastive alignment | `J2_contrastive_{mlp,pooling}/`, `_lam02_ch6/J1_*` | `…/Hybrid-Contrastive/J2_*`, `J1_*_lam02` |
| CTC-conditioned alignment heuristics | `K1_ctc_mse/`, `K2_ctc_posterior/` | `…/Hybrid/{K1_ctc_mse,K2_ctc_posterior}/` |
| Boundary controls (ByT5, Conformer) | `M1_byt5_hybrid_mlp/`, `N1_conformer_hybrid_mlp/` | `…/byt5-small/…`, `…/Hybrid/N1_conformer_hybrid_mlp/` |

`REPRODUCIBILITY.md` lists every row with its exact result path and the rules for
superseded or excluded runs (`K4_sea_contrastive`, pre-fix `J1_*` base runs).

## Exploratory and legacy (retained for provenance, not reported)

`legacy/` (early single-file configs: T5/ByT5 `t5-small-*`, first GPT-2 VLM runs,
hybrid decoder-CTC variants, pretraining and test configs), `experiments/`,
`t5_final/`, `CTC_Primary/`, `AR-InputCorruption*-Equations`, `zero_shot/`,
`AR-SS-Delayed*`, `AR-NoTeacherForcing/`, `AR-ScheduledSamplingFixed*/`
(HWRFormer-L era), `HybridInputCorruption/`, `HybridInputCorruption-XS/`
(λ ≠ 0.1), `H2_*`, `H3_*`, `I1_*`, `K3_ec_loss/`, `K4_sea_contrastive/` (excluded,
see REPRODUCIBILITY.md), `K5_kv_multiview/`, `F1_frozen_enc_vlm/`,
`F2_pretrained_enc/`, `vlm_followup/`, `O1_sentence_eval/`, `others/`.

The `tokenizer:` block and `use_bpe: false` present in most configs are inert:
all thesis models use the character vocabulary in `categories`.
