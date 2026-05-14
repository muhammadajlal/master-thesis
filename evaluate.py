import argparse
import json
import os
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.flop_counter import FlopCounterMode
# Legacy counter retained for fallback; see _LEGACY_get_macs_params() at the
# bottom of this file. To switch back, replace the call in main() and uncomment
# the line below.
# from thop import profile

from rewi.model import BaseModel, DualHeadModel, build_encoder
from rewi.model.multimodal_lm_model import MultimodalLMModel
from rewi.model.pretrainedLM import LMConfig
from rewi.model.vlm_model import VLMModel
from rewi.tokenizer import BPETokenizer

import time


class InferWrapper(nn.Module):
    """Routes the input through the model's inference path.

    For AR-style decoders we call ``model.generate(x, len_max=...)`` so the
    FLOP counter sees the full ``len_max``-step decoding trajectory rather
    than a single forward pass. For non-AR decoders (CTC) we just call
    ``model(x)``. For VLM/LM wrappers we use their own ``generate(x, len_x)``
    method. For DualHeadModel we drive the AR head via its inner
    ``BaseModel.generate``.
    """

    def __init__(self, base_model: nn.Module, mode: str, *, len_max: int = 32) -> None:
        super().__init__()
        self.model = base_model
        self.mode = mode  # one of: "ar", "ctc", "dual", "vlm", "lm"
        self.len_max = int(len_max)

    def forward(self, x: torch.Tensor, len_x: torch.Tensor | None = None) -> torch.Tensor:
        if self.mode in ("ar", "dual"):
            # Both BaseModel(AR) and DualHeadModel expose .generate(x, len_max).
            return self.model.generate(x, len_max=self.len_max)
        if self.mode == "ctc":
            return self.model(x)
        if self.mode in ("vlm", "lm"):
            if len_x is None:
                len_x = torch.tensor([x.size(-1)], dtype=torch.long, device=x.device)
            return self.model.generate(x, len_x)
        raise ValueError(f"unknown InferWrapper mode: {self.mode}")


def get_mean_std_cv(cfgs: dict, results: dict = {}) -> dict:
    '''Calculate the mean and standard deviation of the results of cross
    validation.

    Args:
        cfgs (dict): Configurations.
        results (dict, optional): Current results. Defaults to {}.

    Returns:
        dict: Updated results.
    '''
    cer, wer = {}, {}

    # Max epoch cap (inclusive): restrict "best" search to the trained regime.
    # Resumes and accidental overshoot past cfgs['epoch'] must not leak in.
    max_epoch_inclusive = int(cfgs.get('epoch', 300)) - 1

    def _infer_primary_head_from_cfg(cfg: dict) -> str:
        dual = cfg.get("dual_head", {}) or {}
        if not bool(dual.get("enabled", False)):
            return "ar"

        primary_raw = dual.get("primary", dual.get("primary_head", None))
        primary = str(primary_raw).lower() if primary_raw is not None else "auto"
        if primary in {"", "none", "null"}:
            primary = "auto"
        if primary in {"ar", "ctc"}:
            return primary

        # Auto: infer from the configured loss weights (use schedule.max when enabled)
        try:
            lambda_ar_cfg = float(dual.get("lambda_ar", 1.0))
        except Exception:
            lambda_ar_cfg = 1.0
        try:
            lambda_ctc_cfg = float(dual.get("lambda_ctc", 0.0))
        except Exception:
            lambda_ctc_cfg = 0.0

        sched = dual.get("lambda_ctc_schedule", None) or {}
        sched_enabled = bool(sched.get("enabled", False))
        if sched_enabled and "max" in sched:
            try:
                lambda_ctc_ref = float(sched.get("max", lambda_ctc_cfg))
            except Exception:
                lambda_ctc_ref = lambda_ctc_cfg
        else:
            lambda_ctc_ref = lambda_ctc_cfg

        return "ctc" if lambda_ctc_ref >= lambda_ar_cfg else "ar"

    def _pick_test_epoch_key(result_fd: dict) -> str:
        # Backward compatible with older logs that used epoch=-1 for test.
        if "-1" in result_fd:
            return "-1"
        if "0" in result_fd:
            return "0"
        # Fall back to the smallest int-like epoch key.
        int_keys = []
        for k in result_fd.keys():
            try:
                int_keys.append(int(k))
            except Exception:
                continue
        if int_keys:
            return str(min(int_keys))
        return "0"

    primary = _infer_primary_head_from_cfg(cfgs)
    eval_key = 'evaluation_ctc' if primary == 'ctc' else 'evaluation'

    def _merge_fold_epoch_evals(fold_dir: str):
        """Merge all train_*.json in fold_dir, return dict ep->(cer,wer)
        for epochs <= max_epoch_inclusive. Newer files win on duplicate eps."""
        merged = {}
        files = sorted(
            glob(os.path.join(fold_dir, 'train_*.json')),
            key=lambda p: os.path.getmtime(p),
        )
        for fp in files:
            try:
                with open(fp, 'r') as f:
                    d = json.load(f)
            except Exception:
                continue
            for k, v in d.items():
                if not (isinstance(k, str) and k.isdigit()):
                    continue
                ep = int(k)
                if ep > max_epoch_inclusive:
                    continue
                if not isinstance(v, dict):
                    continue
                ev = v.get(eval_key) or v.get('evaluation')
                if not isinstance(ev, dict):
                    continue
                c = ev.get('character_error_rate')
                w = ev.get('word_error_rate')
                if c is None or w is None:
                    continue
                merged[ep] = (float(c), float(w))
        return merged

    if cfgs['test']:
        # Test mode keeps legacy per-file behavior (test_*.json have a single
        # synthetic epoch key).
        paths_result = glob(
            os.path.join(cfgs['dir_work'], '**', 'test_*.json'),
            recursive=True,
        )
        for i, path_result in enumerate(sorted(paths_result)):
            with open(path_result, 'r') as f:
                result_fd = json.load(f)
            ep = _pick_test_epoch_key(result_fd)
            result_best = result_fd.get(ep, {}).get(
                eval_key, result_fd.get(ep, {}).get('evaluation')
            )
            cer[str(i)] = result_best['character_error_rate']
            wer[str(i)] = result_best['word_error_rate']
    else:
        # Training mode: discover fold directories (each fold is a dir
        # containing one or more train_*.json files from resubmits/resumes).
        fold_dirs = sorted(
            {
                os.path.dirname(p)
                for p in glob(
                    os.path.join(cfgs['dir_work'], '**', 'train_*.json'),
                    recursive=True,
                )
            }
        )
        for i, fold_dir in enumerate(fold_dirs):
            merged = _merge_fold_epoch_evals(fold_dir)
            if not merged:
                print(f"[WARN] no evaluation <= epoch {max_epoch_inclusive} in {fold_dir}")
                continue
            best_ep = min(merged, key=lambda e: merged[e][0])
            c, w = merged[best_ep]
            cer[str(i)] = c
            wer[str(i)] = w

    if cer:
        results['cer'] = {
            'raw': cer,
            'mean': np.mean(list(cer.values())).item(),
            'std': np.std(list(cer.values())).item(),
        }
        results['wer'] = {
            'raw': wer,
            'mean': np.mean(list(wer.values())).item(),
            'std': np.std(list(wer.values())).item(),
        }
        results = {k: v for k, v in sorted(results.items())}

    return results

"""def get_mean_std_cv(cfgs: dict, results: dict = {}) -> dict:
    '''Calculate the mean and standard deviation of the results of cross
    validation.

    Args:
        cfgs (dict): Configurations.
        results (dict, optional): Current results. Defaults to {}.

    Returns:
        dict: Updated results.
    '''
    cer, wer = {}, {}

    if paths_result := glob(
        os.path.join(
            cfgs['dir_work'],
            '*',
            'test_*.json' if cfgs['test'] else 'train_*.json',
        )
    ):
        for i, path_result in enumerate(sorted(paths_result)):
            with open(path_result, 'r') as f:
                result_fd = json.load(f)

            if cfgs['test']:
                result_best = result_fd['-1']['evaluation']
            else:
                epoch_best = result_fd['best']['character_error_rate'][0]
                result_best = result_fd[str(epoch_best)]['evaluation']

            cer[str(i)] = result_best['character_error_rate']
            wer[str(i)] = result_best['word_error_rate']

        results['cer'] = {
            'raw': cer,
            'mean': np.mean(list(cer.values())).item(),
            'std': np.std(list(cer.values())).item(),
        }
        results['wer'] = {
            'raw': wer,
            'mean': np.mean(list(wer.values())).item(),
            'std': np.std(list(wer.values())).item(),
        }
        results = {k: v for k, v in sorted(results.items())}

    return results"""


def get_macs_params(cfgs: dict, results: dict = {}) -> dict:
    '''Calcualte the number of parameters and multiply-accumulate operations
    of the network.

    Args:
        cfgs (dict): Configurations.
        results (dict, optional): Current results. Defaults to {}.

    Returns:
        dict: Updated results.
    '''
    def _infer_vocab_and_ids(cfgs: dict):
        """Mirror main.py's vocab/special-token layout logic.

        Returns:
            vocab_dec, vocab_ctc, pad_id, bos_id, eos_id
        """
        base = int(len(cfgs['categories']))
        arch_de = str(cfgs.get('arch_de'))
        ar_mode = arch_de in {"ar_transformer_xs", "ar_transformer_s", "ar_transformer_m", "ar_transformer_l"}

        if bool(cfgs.get('use_bpe', False)):
            tok_cfg = cfgs.get('tokenizer', {}) or {}
            model_path = tok_cfg.get('model', None)
            if not model_path:
                raise ValueError("use_bpe=true but tokenizer.model is missing in config")
            tok = BPETokenizer(model_path)
            vocab_dec = int(tok.vocab_size)
            pad_id, bos_id, eos_id = int(tok.PAD), int(tok.BOS), int(tok.EOS)
            vocab_ctc = base
            return vocab_dec, vocab_ctc, pad_id, bos_id, eos_id

        # Character mode
        pad_id, bos_id, eos_id = base, base + 1, base + 2
        vocab_dec = base + (3 if ar_mode else 0)
        vocab_ctc = base
        return int(vocab_dec), int(vocab_ctc), int(pad_id), int(bos_id), int(eos_id)

    arch_de = str(cfgs.get('arch_de', ''))
    lm_mode = arch_de in {'byt5_small', 't5-small'}
    vlm_mode = bool(cfgs.get('vlm_enabled', False)) or arch_de == 'vlm'

    if vlm_mode:
        # Build encoder + VLM wrapper mirroring main.py
        encoder = build_encoder(cfgs['num_channel'], cfgs['arch_en'], cfgs['len_seq'])
        ratio_ds = int(getattr(encoder, 'ratio_ds', 1))
        d_cnn = int(cfgs.get('d_cnn', 512))

        vlm_cfg = cfgs.get('vlm', {}) or {}
        lm_name = str(vlm_cfg.get('lm_name', 'gpt2'))

        categories = cfgs.get('categories', None)
        vocab_ctc = int(len(categories)) if (categories and bool(vlm_cfg.get('hybrid_ctc', False))) else None

        model = VLMModel(
            encoder=encoder,
            ratio_ds=ratio_ds,
            d_cnn=d_cnn,
            lm_name_or_path=lm_name,
            connector_type=str(vlm_cfg.get('connector_type', 'qformer')),
            num_queries=int(vlm_cfg.get('num_queries', 32)),
            qformer_layers=int(vlm_cfg.get('qformer_layers', 4)),
            qformer_nhead=int(vlm_cfg.get('qformer_nhead', 8)),
            qformer_dropout=float(vlm_cfg.get('qformer_dropout', 0.1)),
            prompt_text=vlm_cfg.get('prompt_text', 'Transcribe the handwritten text from IMU sensor signals:'),
            refine_prompt_text=vlm_cfg.get('refine_prompt_text', None),
            two_step_decode=bool(vlm_cfg.get('two_step_decode', False)),
            num_soft_tokens=int(vlm_cfg.get('num_soft_tokens', 20)),
            freeze_encoder=bool(cfgs.get('freeze', False)),
            freeze_lm=bool(vlm_cfg.get('freeze_lm', True)),
            use_lora=bool(vlm_cfg.get('use_lora', False)),
            lora_r=int(vlm_cfg.get('lora_r', 16)),
            lora_alpha=int(vlm_cfg.get('lora_alpha', 32)),
            lora_dropout=float(vlm_cfg.get('lora_dropout', 0.05)),
            lora_target_modules=vlm_cfg.get('lora_target_modules', None),
            max_new_tokens=int(vlm_cfg.get('max_new_tokens', 64)),
            num_beams=int(vlm_cfg.get('num_beams', 1)),
            repetition_penalty=float(vlm_cfg.get('repetition_penalty', 1.0)),
            local_files_only=bool(vlm_cfg.get('local_files_only', True)),
            z_dropout=float(vlm_cfg.get('z_dropout', 0.1)),
            label_smoothing=float(vlm_cfg.get('label_smoothing', 0.0)),
            vocab_ctc=vocab_ctc,
            categories=categories,
        ).eval()
        wrapper_mode = "vlm"

    elif lm_mode:
        # Build encoder + LM wrapper mirroring main.py
        encoder = build_encoder(cfgs['num_channel'], cfgs['arch_en'], cfgs['len_seq'])
        ratio_ds = int(getattr(encoder, 'ratio_ds', 1))

        lm_cfg = LMConfig(
            name=cfgs.get('lm_name', 'google/byt5-small'),
            train_lm=bool(cfgs.get('lm_train_lm', False)),
            max_new_tokens=int(cfgs.get('lm_max_new_tokens', 128)),
            num_beams=int(cfgs.get('lm_num_beams', 1)),
            length_penalty=float(cfgs.get('lm_length_penalty', 1.0)),
            min_new_tokens=int(cfgs.get('lm_min_new_tokens', 0)),
            local_files_only=bool(cfgs.get('lm_local_files_only', True)),
        )

        d_cnn = int(cfgs.get('d_cnn', 0))
        model = MultimodalLMModel(
            encoder=encoder,
            ratio_ds=ratio_ds,
            d_cnn=d_cnn,
            lm_cfg=lm_cfg,
            proj_dropout=float(cfgs.get('lm_proj_dropout', 0.0)),
            freeze_encoder=bool(cfgs.get('freeze', True)),
        ).eval()
        wrapper_mode = "lm"

    else:
        vocab_dec, vocab_ctc, pad_id, bos_id, eos_id = _infer_vocab_and_ids(cfgs)

        # Check if hybrid (dual-head) mode
        dual_cfg = cfgs.get('dual_head', {}) or {}
        dual_enabled = isinstance(dual_cfg, dict) and bool(dual_cfg.get('enabled', False))

        if dual_enabled:
            if bool(cfgs.get('use_bpe', False)):
                raise ValueError("dual_head.enabled currently supports character mode only (use_bpe must be false)")

            tie_cfg = dual_cfg.get('tie', {}) or {}
            model = DualHeadModel(
                arch_en=cfgs['arch_en'],
                arch_ar=cfgs['arch_de'],
                arch_ctc=str(dual_cfg.get('arch_ctc', 'linear')),
                in_chan=cfgs['num_channel'],
                vocab_ar=vocab_dec,
                vocab_ctc=vocab_ctc,
                len_seq=cfgs['len_seq'],
                use_gated_attention=bool(cfgs.get('use_gated_attention', False)),
                gating_type=str(cfgs.get('gating_type', 'elementwise')),
                pad_id=pad_id,
                bos_id=bos_id,
                eos_id=eos_id,
                tie_cfg=tie_cfg,
                dual_cfg=dual_cfg,
            ).eval()
            wrapper_mode = "dual"
        else:
            model = BaseModel(
                cfgs['arch_en'],
                cfgs['arch_de'],
                cfgs['num_channel'],
                vocab_dec,
                cfgs['len_seq'],
                use_gated_attention=bool(cfgs.get('use_gated_attention', False)),
                gating_type=str(cfgs.get('gating_type', 'elementwise')),
                vocab_ctc=vocab_ctc,
                pad_id=pad_id,
                decoder_side_ctc_cfg=cfgs.get('decoder_side_ctc', {}) or {},
            ).eval()
            wrapper_mode = "ar" if getattr(model, "use_ar", False) else "ctc"

        # BaseModel / DualHeadModel both expose .infer() to switch the encoder
        # and AR decoder to their inference-mode fast paths (fused BN, cached attn).
        model.infer()

    # ---- Common: build dummy input, wrap, count FLOPs and total params ------
    # Following the supervisor's approach (FlopCounterMode is ATen-level and
    # captures attention SDPA + HF Transformer internals that THOP missed;
    # InferWrapper.generate() runs the full autoregressive decode rather than
    # a single forward step). For VLM/LM the model has its own generate(x,len_x).
    T = 1024 if 'word' in cfgs['dir_dataset'] else 4096
    x = torch.randn(1, cfgs['num_channel'], T)
    len_x = torch.tensor([T], dtype=torch.long)
    len_max = int(cfgs.get('generate_len_max', 32))

    wrapper = InferWrapper(model, mode=wrapper_mode, len_max=len_max)

    # Params: total count including frozen branches (robust across thop's blind spots).
    params = sum(p.numel() for p in model.parameters())

    flops = 0
    try:
        with torch.no_grad():
            with FlopCounterMode(wrapper, display=False) as flop_counter:
                if wrapper_mode in ("vlm", "lm"):
                    wrapper(x, len_x=len_x)
                else:
                    wrapper(x)
            flops = int(flop_counter.get_total_flops())
    except Exception as exc:
        print(f"[WARN] FlopCounterMode failed for mode={wrapper_mode}: {exc}")
        flops = 0

    results['params'] = int(params)
    results['flops'] = int(flops)
    # Legacy field: 1 MAC ≈ 2 FLOPs (one multiply + one add).
    results['macs'] = int(flops // 2)
    results['generate_len_max'] = len_max
    results = {k: v for k, v in sorted(results.items())}
    return results


def main(path_cfg: str) -> None:
    '''Main function.

    Args:
        path_cfg (str): Path to the configuration YAML file.
    '''
    with open(path_cfg, 'r') as f:
        cfgs = yaml.safe_load(f)

    os.makedirs(cfgs['dir_work'], exist_ok=True)

    if os.path.isfile(os.path.join(cfgs['dir_work'], 'results.json')):
        with open(os.path.join(cfgs['dir_work'], 'results.json'), 'r') as f:
            results = json.load(f)
    else:
        results = {}

    results = get_mean_std_cv(cfgs, results)
    results = get_macs_params(cfgs, results)

    with open(os.path.join(cfgs['dir_work'], 'results.json'), 'w') as f:
        json.dump(results, f)

    print(results)


def main_ac(dir_work: str) -> None:
    '''Summarize the results of cross-dataset evaluation.

    Args:
        dir_work (str): Path to the work directory.
    '''
    cer, wer = {'raw': {}}, {'raw': {}}

    for fname in glob(os.path.join(dir_work, '*', 'results.json')):
        with open(fname, 'r') as f:
            result = json.load(f)

        idx_1 = os.path.basename(os.path.dirname(fname))

        for idx_2 in ['0', '1', '2', '3', '4']:
            cer['raw'][f'{idx_1}{idx_2}'] = result['cer']['raw'][idx_2]
            wer['raw'][f'{idx_1}{idx_2}'] = result['wer']['raw'][idx_2]

    cer['mean'] = np.mean(list(cer['raw'].values())).item()
    cer['std'] = np.std(list(cer['raw'].values())).item()
    wer['mean'] = np.mean(list(wer['raw'].values())).item()
    wer['std'] = np.std(list(wer['raw'].values())).item()
    results = {'cer': cer, 'wer': wer}

    with open(os.path.join(dir_work, 'results.json'), 'w') as f:
        json.dump(results, f)

    print(results)


# =============================================================================
# LEGACY: pre-2026-05-11 thop-based compute counter.
#
# Switched to FlopCounterMode + InferWrapper(generate) on 2026-05-11 because
# `thop.profile` (a) silently undercounts params in `nn.MultiheadAttention`
# and gated cross-attention (our AR baseline was reported as 4.82M but is
# actually 6.94M), and (b) only profiles a single forward step rather than
# the full autoregressive generation, so reported MACs for AR/Hybrid/VLM/LM
# were significantly under-reported.
#
# Snapshot of all results.json numbers produced by this legacy code is at
# `results/compute_old_thop.{csv,json}` (captured 2026-05-11).
#
# To fall back:
#   1) Uncomment `from thop import profile` at the top of this file.
#   2) Rename `_LEGACY_get_macs_params` to `get_macs_params` (replacing the
#      FlopCounterMode-based implementation above).
# =============================================================================
#
# def _LEGACY_get_macs_params(cfgs: dict, results: dict = {}) -> dict:
#     '''Calcualte the number of parameters and multiply-accumulate operations
#     of the network.
#
#     LEGACY thop-based implementation. See header note above for why it was
#     replaced.
#     '''
#     def _infer_vocab_and_ids(cfgs: dict):
#         """Mirror main.py's vocab/special-token layout logic."""
#         base = int(len(cfgs['categories']))
#         arch_de = str(cfgs.get('arch_de'))
#         ar_mode = arch_de in {"ar_transformer_xs", "ar_transformer_s", "ar_transformer_m", "ar_transformer_l"}
#
#         if bool(cfgs.get('use_bpe', False)):
#             tok_cfg = cfgs.get('tokenizer', {}) or {}
#             model_path = tok_cfg.get('model', None)
#             if not model_path:
#                 raise ValueError("use_bpe=true but tokenizer.model is missing in config")
#             tok = BPETokenizer(model_path)
#             vocab_dec = int(tok.vocab_size)
#             pad_id, bos_id, eos_id = int(tok.PAD), int(tok.BOS), int(tok.EOS)
#             vocab_ctc = base
#             return vocab_dec, vocab_ctc, pad_id, bos_id, eos_id
#
#         # Character mode
#         pad_id, bos_id, eos_id = base, base + 1, base + 2
#         vocab_dec = base + (3 if ar_mode else 0)
#         vocab_ctc = base
#         return int(vocab_dec), int(vocab_ctc), int(pad_id), int(bos_id), int(eos_id)
#
#     class _ProfileWrapper(torch.nn.Module):
#         """Force execution of optional branches so THOP counts their params."""
#
#         def __init__(self, model, *, kind: str, force_dec_ctc: bool = False,
#                      force_deep_supervision: bool = False) -> None:
#             super().__init__()
#             self.model = model
#             self.kind = str(kind)
#             self.force_dec_ctc = bool(force_dec_ctc)
#             self.force_deep_supervision = bool(force_deep_supervision)
#
#         def forward(self, x):
#             with torch.no_grad():
#                 if self.kind == 'dual':
#                     force_train = bool(self.force_dec_ctc or self.force_deep_supervision)
#                     prev_mode = self.model.training
#                     try:
#                         self.model.train(force_train)
#                         B = x.size(0)
#                         y_inp = torch.zeros(B, 2, dtype=torch.long, device=x.device)
#                         out = self.model(x, y_inp=y_inp, return_ar=True, return_ctc=True)
#                         y = out.get('ar_logits', None)
#                         if y is None:
#                             y = out['ctc_logits']
#                         for t in out.get('ar_logits_layers', []) or []:
#                             y = y + (t.sum() * 0.0)
#                         for t in out.get('dec_ctc_logits_layers', []) or []:
#                             y = y + (t.sum() * 0.0)
#                         return y
#                     finally:
#                         self.model.train(prev_mode)
#
#                 if self.kind == 'base':
#                     B = x.size(0)
#                     y_inp = torch.zeros(B, 2, dtype=torch.long, device=x.device)
#                     out = self.model(x, y_inp=y_inp,
#                                      return_ar_layers=bool(self.force_dec_ctc),
#                                      return_dec_ctc=bool(self.force_dec_ctc))
#                     if isinstance(out, dict):
#                         y = out['logits']
#                         for t in out.get('dec_ctc_logits_layers', []) or []:
#                             y = y + (t.sum() * 0.0)
#                         for t in out.get('logits_layers', []) or []:
#                             y = y + (t.sum() * 0.0)
#                         return y
#                     return out
#
#                 return self.model(x)
#
#     arch_de = str(cfgs.get('arch_de', ''))
#     lm_mode = arch_de in {'byt5_small', 't5-small'}
#     vlm_mode = bool(cfgs.get('vlm_enabled', False)) or arch_de == 'vlm'
#
#     if vlm_mode:
#         encoder = build_encoder(cfgs['num_channel'], cfgs['arch_en'], cfgs['len_seq'])
#         ratio_ds = int(getattr(encoder, 'ratio_ds', 1))
#         d_cnn = int(cfgs.get('d_cnn', 512))
#         vlm_cfg = cfgs.get('vlm', {}) or {}
#         lm_name = str(vlm_cfg.get('lm_name', 'gpt2'))
#         categories = cfgs.get('categories', None)
#         vocab_ctc = int(len(categories)) if (categories and bool(vlm_cfg.get('hybrid_ctc', False))) else None
#
#         model = VLMModel(
#             encoder=encoder, ratio_ds=ratio_ds, d_cnn=d_cnn, lm_name_or_path=lm_name,
#             connector_type=str(vlm_cfg.get('connector_type', 'qformer')),
#             num_queries=int(vlm_cfg.get('num_queries', 32)),
#             qformer_layers=int(vlm_cfg.get('qformer_layers', 4)),
#             qformer_nhead=int(vlm_cfg.get('qformer_nhead', 8)),
#             qformer_dropout=float(vlm_cfg.get('qformer_dropout', 0.1)),
#             prompt_text=vlm_cfg.get('prompt_text', 'Transcribe the handwritten text from IMU sensor signals:'),
#             refine_prompt_text=vlm_cfg.get('refine_prompt_text', None),
#             two_step_decode=bool(vlm_cfg.get('two_step_decode', False)),
#             num_soft_tokens=int(vlm_cfg.get('num_soft_tokens', 20)),
#             freeze_encoder=bool(cfgs.get('freeze', False)),
#             freeze_lm=bool(vlm_cfg.get('freeze_lm', True)),
#             use_lora=bool(vlm_cfg.get('use_lora', False)),
#             lora_r=int(vlm_cfg.get('lora_r', 16)),
#             lora_alpha=int(vlm_cfg.get('lora_alpha', 32)),
#             lora_dropout=float(vlm_cfg.get('lora_dropout', 0.05)),
#             lora_target_modules=vlm_cfg.get('lora_target_modules', None),
#             max_new_tokens=int(vlm_cfg.get('max_new_tokens', 64)),
#             num_beams=int(vlm_cfg.get('num_beams', 1)),
#             repetition_penalty=float(vlm_cfg.get('repetition_penalty', 1.0)),
#             local_files_only=bool(vlm_cfg.get('local_files_only', True)),
#             z_dropout=float(vlm_cfg.get('z_dropout', 0.1)),
#             label_smoothing=float(vlm_cfg.get('label_smoothing', 0.0)),
#             vocab_ctc=vocab_ctc, categories=categories,
#         ).eval()
#
#         T = 1024 if 'word' in cfgs['dir_dataset'] else 4096
#         x = torch.randn(1, cfgs['num_channel'], T)
#         len_x = torch.tensor([T], dtype=torch.long)
#         labels = torch.zeros((1, 8), dtype=torch.long)
#         try:
#             macs, params = profile(model, inputs=(x, len_x, labels), verbose=False)
#         except Exception as exc:
#             params = sum(p.numel() for p in model.parameters())
#             macs = 0
#             print(f"[WARN] THOP MACs failed for VLM; returning macs=0. Error: {exc}")
#
#     elif lm_mode:
#         encoder = build_encoder(cfgs['num_channel'], cfgs['arch_en'], cfgs['len_seq'])
#         ratio_ds = int(getattr(encoder, 'ratio_ds', 1))
#         lm_cfg = LMConfig(
#             name=cfgs.get('lm_name', 'google/byt5-small'),
#             train_lm=bool(cfgs.get('lm_train_lm', False)),
#             max_new_tokens=int(cfgs.get('lm_max_new_tokens', 128)),
#             num_beams=int(cfgs.get('lm_num_beams', 1)),
#             length_penalty=float(cfgs.get('lm_length_penalty', 1.0)),
#             min_new_tokens=int(cfgs.get('lm_min_new_tokens', 0)),
#             local_files_only=bool(cfgs.get('lm_local_files_only', True)),
#         )
#         d_cnn = int(cfgs.get('d_cnn', 0))
#         model = MultimodalLMModel(
#             encoder=encoder, ratio_ds=ratio_ds, d_cnn=d_cnn, lm_cfg=lm_cfg,
#             proj_dropout=float(cfgs.get('lm_proj_dropout', 0.0)),
#             freeze_encoder=bool(cfgs.get('freeze', True)),
#         ).eval()
#         T = 1024 if 'word' in cfgs['dir_dataset'] else 4096
#         x = torch.randn(1, cfgs['num_channel'], T)
#         len_x = torch.tensor([T], dtype=torch.long)
#         labels = torch.zeros((1, 8), dtype=torch.long)
#         macs, params = profile(model, inputs=(x, len_x, labels), verbose=False)
#
#     else:
#         vocab_dec, vocab_ctc, pad_id, bos_id, eos_id = _infer_vocab_and_ids(cfgs)
#         dual_cfg = cfgs.get('dual_head', {}) or {}
#         dual_enabled = isinstance(dual_cfg, dict) and bool(dual_cfg.get('enabled', False))
#
#         if dual_enabled:
#             if bool(cfgs.get('use_bpe', False)):
#                 raise ValueError("dual_head.enabled currently supports character mode only")
#             tie_cfg = dual_cfg.get('tie', {}) or {}
#             model = DualHeadModel(
#                 arch_en=cfgs['arch_en'], arch_ar=cfgs['arch_de'],
#                 arch_ctc=str(dual_cfg.get('arch_ctc', 'linear')),
#                 in_chan=cfgs['num_channel'], vocab_ar=vocab_dec, vocab_ctc=vocab_ctc,
#                 len_seq=cfgs['len_seq'],
#                 use_gated_attention=bool(cfgs.get('use_gated_attention', False)),
#                 gating_type=str(cfgs.get('gating_type', 'elementwise')),
#                 pad_id=pad_id, bos_id=bos_id, eos_id=eos_id,
#                 tie_cfg=tie_cfg, dual_cfg=dual_cfg,
#             ).eval()
#         else:
#             model = BaseModel(
#                 cfgs['arch_en'], cfgs['arch_de'], cfgs['num_channel'],
#                 vocab_dec, cfgs['len_seq'],
#                 use_gated_attention=bool(cfgs.get('use_gated_attention', False)),
#                 gating_type=str(cfgs.get('gating_type', 'elementwise')),
#                 vocab_ctc=vocab_ctc, pad_id=pad_id,
#                 decoder_side_ctc_cfg=cfgs.get('decoder_side_ctc', {}) or {},
#             ).eval()
#
#         model.infer()
#         T = 1024 if 'word' in cfgs['dir_dataset'] else 4096
#         x = torch.randn(1, cfgs['num_channel'], T)
#
#         if dual_enabled:
#             dec_ctc_cfg = dual_cfg.get('decoder_side_ctc', {}) or {}
#             deep_cfg = dual_cfg.get('deep_supervision', {}) or {}
#             wrap = _ProfileWrapper(model, kind='dual',
#                                    force_dec_ctc=bool(dec_ctc_cfg.get('enabled', False)),
#                                    force_deep_supervision=bool(deep_cfg.get('enabled', False)))
#             macs, params = profile(wrap, inputs=(x,), verbose=False)
#         else:
#             dec_ctc_cfg = cfgs.get('decoder_side_ctc', {}) or {}
#             wrap = _ProfileWrapper(model, kind='base',
#                                    force_dec_ctc=bool(dec_ctc_cfg.get('enabled', False)))
#             macs, params = profile(wrap, inputs=(x,), verbose=False)
#
#     results['macs'] = int(macs)
#     results['params'] = int(params)
#     results = {k: v for k, v in sorted(results.items())}
#     return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate handwriting recognition model.'
    )
    parser.add_argument(
        '-c', '--config', help='Path to YAML file of configuration.'
    )
    args = parser.parse_args()

    if os.path.isfile(args.config):
        main(args.config)
    elif os.path.isdir(args.config):
        main_ac(args.config)
