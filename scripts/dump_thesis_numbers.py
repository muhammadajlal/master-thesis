#!/usr/bin/env python3
"""Dump CER/WER means for active thesis experiments and excluded provenance runs."""
import json, os, sys

ROOT = "/home/woody/iwso/iwso214h/imu-hwr/results/hwr2"

# (label, relative path)
EXPS = [
    # classical baselines (Stabilo word/sent, OnHW WI)
    ("REWI Stabilo word",  "Baseline-REWI/bilstm_wide__wi_word_hw6_meta"),
    ("REWI Stabilo sent",  "Baseline-REWI/bilstm_wide__wi_sent_hw6_meta"),
    ("REWI OnHW WI",       "Baseline-REWI/bilstm_wide__onhw_wi_word_rh"),
    ("REWI OnHW WD",       "Baseline-REWI/bilstm_wide__onhw_wd_word_rh"),
    ("TransCTC OnHW WI",   "Baseline-Transformer-CTC/transformer_s__onhw_wi_word_rh"),
    ("AR-Ungated Stabilo word", None),  # not present
    ("AR-Ungated OnHW WI", "Baseline-AR-Ungated/ar_transformer_s__onhw_wi_word_rh"),
    ("AR-ElemGat Stabilo word", "Baseline-AR-ElementwiseGating/ar_transformer_s__wi_word_hw6_meta"),
    ("AR-ElemGat Stabilo sent", "Baseline-AR-ElementwiseGating/ar_transformer_s__wi_sent_hw6_meta"),
    ("AR-ElemGat OnHW WI", "Baseline-AR-ElementwiseGating/ar_transformer_s__onhw_wi_word_rh"),
    ("AR-HeadGat Stabilo word", "Baseline-AR-HeadwiseGating/ar_transformer_s__wi_word_hw6_meta"),
    ("AR-HeadGat Stabilo sent", "Baseline-AR-HeadwiseGating/ar_transformer_s__wi_sent_hw6_meta"),
    ("AR-HeadGat OnHW WI", "Baseline-AR-HeadwiseGating/ar_transformer_s__onhw_wi_word_rh"),
    # VLM phase 1
    ("A0 Qformer OnHW",  "Ablations-MMLM/GPT-2/AR-only/vlm_Qformer_gpt2_word_v2/vlm__onhw_wi_word_rh"),
    ("A0 Qformer Stabilo", "Ablations-MMLM/GPT-2/AR-only/vlm_Qformer_gpt2_word_v2/vlm__wi_word_hw6_meta"),
    ("A1 MLP OnHW",  "Ablations-MMLM/GPT-2/AR-only/vlm_ablation_A1_mlp_pretrained/vlm__onhw_wi_word_rh"),
    ("A1 MLP Stabilo","Ablations-MMLM/GPT-2/AR-only/vlm_ablation_A1_mlp_pretrained/vlm__wi_word_hw6_meta"),
    ("A2 random OnHW","Ablations-MMLM/GPT-2/AR-only/vlm_ablation_A2_mlp_random/vlm__onhw_wi_word_rh"),
    ("A2 random Stabilo","Ablations-MMLM/GPT-2/AR-only/vlm_ablation_A2_mlp_random/vlm__wi_word_hw6_meta"),
    ("A3 Linear OnHW","Ablations-MMLM/GPT-2/AR-only/vlm_ablation_A3_linear_pretrained/vlm__onhw_wi_word_rh"),
    ("A3 Linear Stabilo","Ablations-MMLM/GPT-2/AR-only/vlm_ablation_A3_linear_pretrained/vlm__wi_word_hw6_meta"),
    ("B1 Pooling OnHW","Ablations-MMLM/GPT-2/AR-only/vlm_ablation_B1_pooling_pretrained/vlm__onhw_wi_word_rh"),
    ("B1 Pooling Stabilo","Ablations-MMLM/GPT-2/AR-only/vlm_ablation_B1_pooling_pretrained/vlm__wi_word_hw6_meta"),
    # Phase 2
    ("G1a Linear OnHW","Ablations-MMLM/GPT-2/AR-only/G1a_linear/vlm__onhw_wi_word_rh"),
    ("G1a Linear Stabilo","Ablations-MMLM/GPT-2/AR-only/G1a_linear/vlm__wi_word_hw6_meta"),
    ("G1b ConvPool OnHW","Ablations-MMLM/GPT-2/AR-only/G1b_conv_pool/vlm__onhw_wi_word_rh"),
    ("G1b ConvPool Stabilo","Ablations-MMLM/GPT-2/AR-only/G1b_conv_pool/vlm__wi_word_hw6_meta"),
    # Phase 3
    ("F1 OnHW","Ablations-MMLM/GPT-2/AR-only/F1_frozen_enc_mlp/vlm__onhw_wi_word_rh"),
    ("F1 Stabilo","Ablations-MMLM/GPT-2/AR-only/F1_frozen_enc_mlp/vlm__wi_word_hw6_meta"),
    ("F2 OnHW","Ablations-MMLM/GPT-2/AR-only/F2_vlm_enc_ar/ar_transformer_s__onhw_wi_word_rh"),
    ("F2 Stabilo","Ablations-MMLM/GPT-2/AR-only/F2_vlm_enc_ar/ar_transformer_s__wi_word_hw6_meta"),
    # Phase 4
    ("H1 MLP OnHW","Ablations-MMLM/GPT-2/Hybrid/H1_hybrid_mlp/vlm__onhw_wi_word_rh"),
    ("H1 MLP Stabilo","Ablations-MMLM/GPT-2/Hybrid/H1_hybrid_mlp/vlm__wi_word_hw6_meta"),
    ("H1p Pool OnHW","Ablations-MMLM/GPT-2/Hybrid/H1_hybrid_pooling/vlm__onhw_wi_word_rh"),
    ("H1p Pool Stabilo","Ablations-MMLM/GPT-2/Hybrid/H1_hybrid_pooling/vlm__wi_word_hw6_meta"),
    ("H2p Promptaug OnHW","Ablations-MMLM/GPT-2/Hybrid/H2_hybrid_prompt_aug_pooling/vlm__onhw_wi_word_rh"),
    ("H2p Promptaug Stabilo","Ablations-MMLM/GPT-2/Hybrid/H2_hybrid_prompt_aug_pooling/vlm__wi_word_hw6_meta"),
    ("H3p Twostep OnHW","Ablations-MMLM/GPT-2/Hybrid/H3_hybrid_two_step_prompt_pooling/vlm__onhw_wi_word_rh"),
    ("H3p Twostep Stabilo","Ablations-MMLM/GPT-2/Hybrid/H3_hybrid_two_step_prompt_pooling/vlm__wi_word_hw6_meta"),
    # Phase 5
    ("J1 Contr MLP OnHW","Ablations-MMLM/GPT-2/Hybrid-Contrastive/J1_contrastive_mlp/vlm__onhw_wi_word_rh"),
    ("J1 Contr MLP Stabilo","Ablations-MMLM/GPT-2/Hybrid-Contrastive/J1_contrastive_mlp/vlm__wi_word_hw6_meta"),
    ("J1 Contr Pool OnHW","Ablations-MMLM/GPT-2/Hybrid-Contrastive/J1_contrastive_pooling/vlm__onhw_wi_word_rh"),
    ("J1 Contr Pool Stabilo","Ablations-MMLM/GPT-2/Hybrid-Contrastive/J1_contrastive_pooling/vlm__wi_word_hw6_meta"),
    ("J2 Contr MLP OnHW","Ablations-MMLM/GPT-2/Hybrid-Contrastive/J2_contrastive_mlp/vlm__onhw_wi_word_rh"),
    ("J2 Contr MLP Stabilo","Ablations-MMLM/GPT-2/Hybrid-Contrastive/J2_contrastive_mlp/vlm__wi_word_hw6_meta"),
    ("J2 Contr Pool OnHW","Ablations-MMLM/GPT-2/Hybrid-Contrastive/J2_contrastive_pooling/vlm__onhw_wi_word_rh"),
    ("J2 Contr Pool Stabilo","Ablations-MMLM/GPT-2/Hybrid-Contrastive/J2_contrastive_pooling/vlm__wi_word_hw6_meta"),
    ("K3 EC OnHW","Ablations-MMLM/GPT-2/Hybrid-Contrastive/K3_ec_loss/vlm__onhw_wi_word_rh"),
    ("K3 EC Stabilo","Ablations-MMLM/GPT-2/Hybrid-Contrastive/K3_ec_loss/vlm__wi_word_hw6_meta"),
    # Phase 6
    ("K1 CTC-MSE OnHW","Ablations-MMLM/GPT-2/Hybrid/K1_ctc_mse/vlm__onhw_wi_word_rh"),
    ("K1 CTC-MSE Stabilo","Ablations-MMLM/GPT-2/Hybrid/K1_ctc_mse/vlm__wi_word_hw6_meta"),
    ("K2 CTC-Posterior OnHW","Ablations-MMLM/GPT-2/Hybrid/K2_ctc_posterior/vlm__onhw_wi_word_rh"),
    ("K2 CTC-Posterior Stabilo","Ablations-MMLM/GPT-2/Hybrid/K2_ctc_posterior/vlm__wi_word_hw6_meta"),
    ("K5 KV-MV OnHW","Ablations-MMLM/GPT-2/Hybrid/K5_kv_multiview/vlm__onhw_wi_word_rh"),
    ("K5 KV-MV Stabilo","Ablations-MMLM/GPT-2/Hybrid/K5_kv_multiview/vlm__wi_word_hw6_meta"),
    # t5
    ("T5 60-300 OnHW", "Ablations-MMLM/t5-small/multimodal_t5_small-60-300/t5-small__onhw_wi_word_rh"),
    ("T5 0-600 OnHW",  "Ablations-MMLM/t5-small/multimodal_t5_small-0-600_word/t5-small__onhw_wi_word_rh"),
    # Historical K4 values are printed for Appendix provenance only, never as active RQ3 evidence.
    ("EXCLUDED K4 OnHW","Ablations-MMLM/GPT-2/Hybrid-Contrastive/K4_sea_contrastive/vlm__onhw_wi_word_rh_lam02"),
    ("EXCLUDED K4 private","Ablations-MMLM/GPT-2/Hybrid-Contrastive/K4_sea_contrastive/vlm__wi_word_hw6_meta"),
]

rows = []
for label, rel in EXPS:
    if rel is None:
        print(f"{label:38s}  -- not present")
        continue
    p = os.path.join(ROOT, rel, "results.json")
    if not os.path.isfile(p):
        print(f"{label:38s}  MISSING {p}")
        continue
    try:
        d = json.load(open(p))
        cer = d["cer"]["mean"] * 100
        wer = d["wer"]["mean"] * 100
        n = len(d["cer"].get("raw", {}))
        macs = d.get("macs")
        params = d.get("params")
        print(f"{label:38s}  CER={cer:6.2f}  WER={wer:6.2f}  n={n}  params={params}")
    except Exception as e:
        print(f"{label:38s}  ERROR {e}")
