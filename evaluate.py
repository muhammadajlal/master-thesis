import argparse
import json
import os
from glob import glob

import numpy as np
import torch
import yaml
from thop import profile

from rewi.model import BaseModel, DualHeadModel, build_encoder
from rewi.model.multimodal_lm_model import MultimodalLMModel
from rewi.model.pretrainedLM import LMConfig
from rewi.tokenizer import BPETokenizer

import time


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

    # ✅ CHANGED: make the glob recursive so it works for nested fold_X/0..4/ layouts
    pattern = 'test_*.json' if cfgs['test'] else 'train_*.json'        # <-- ADDED (pulled out for clarity)
    paths_result = glob(
        os.path.join(cfgs['dir_work'], '**', pattern), recursive=True   # <-- CHANGED '**' + recursive=True
    )

    if paths_result:
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
        ar_mode = arch_de in {"ar_transformer_s", "ar_transformer_m", "ar_transformer_l"}

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

    class _ProfileWrapper(torch.nn.Module):
        """Force execution of optional branches so THOP counts their params."""

        def __init__(
            self,
            model: torch.nn.Module,
            *,
            kind: str,
            force_dec_ctc: bool = False,
            force_deep_supervision: bool = False,
        ) -> None:
            super().__init__()
            self.model = model
            self.kind = str(kind)
            self.force_dec_ctc = bool(force_dec_ctc)
            self.force_deep_supervision = bool(force_deep_supervision)

        def forward(self, x: torch.Tensor):
            with torch.no_grad():
                # DualHeadModel aux branches are gated on model.training.
                # BaseModel dec-CTC is gated on forward(return_dec_ctc=True).
                if self.kind == 'dual':
                    force_train = bool(self.force_dec_ctc or self.force_deep_supervision)
                    prev_mode = self.model.training
                    try:
                        self.model.train(force_train)
                        B = x.size(0)
                        y_inp = torch.zeros(B, 2, dtype=torch.long, device=x.device)
                        out = self.model(x, y_inp=y_inp, return_ar=True, return_ctc=True)
                        y = out.get('ar_logits', None)
                        if y is None:
                            y = out['ctc_logits']
                        for t in out.get('ar_logits_layers', []) or []:
                            y = y + (t.sum() * 0.0)
                        for t in out.get('dec_ctc_logits_layers', []) or []:
                            y = y + (t.sum() * 0.0)
                        return y
                    finally:
                        self.model.train(prev_mode)

                if self.kind == 'base':
                    # Try AR-style forward to ensure decoder-side CTC executes.
                    B = x.size(0)
                    y_inp = torch.zeros(B, 2, dtype=torch.long, device=x.device)
                    out = self.model(
                        x,
                        y_inp=y_inp,
                        return_ar_layers=bool(self.force_dec_ctc),
                        return_dec_ctc=bool(self.force_dec_ctc),
                    )
                    if isinstance(out, dict):
                        y = out['logits']
                        for t in out.get('dec_ctc_logits_layers', []) or []:
                            y = y + (t.sum() * 0.0)
                        for t in out.get('logits_layers', []) or []:
                            y = y + (t.sum() * 0.0)
                        return y
                    return out

                # Fallback: just run the model.
                return self.model(x)

    lm_mode = cfgs.get('arch_de') in {'byt5_small', 't5-small'}

    if lm_mode:
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

        # Dummy inputs for profiling (word vs sentence length heuristic)
        T = 1024 if 'word' in cfgs['dir_dataset'] else 4096
        x = torch.randn(1, cfgs['num_channel'], T)
        len_x = torch.tensor([T], dtype=torch.long)
        labels = torch.zeros((1, 8), dtype=torch.long)  # small label length to keep thop fast

        macs, params = profile(model, inputs=(x, len_x, labels), verbose=False)
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

        model.infer()
        T = 1024 if 'word' in cfgs['dir_dataset'] else 4096
        x = torch.randn(1, cfgs['num_channel'], T)

        if dual_enabled:
            dec_ctc_cfg = dual_cfg.get('decoder_side_ctc', {}) or {}
            deep_cfg = dual_cfg.get('deep_supervision', {}) or {}
            wrap = _ProfileWrapper(
                model,
                kind='dual',
                force_dec_ctc=bool(dec_ctc_cfg.get('enabled', False)),
                force_deep_supervision=bool(deep_cfg.get('enabled', False)),
            )
            macs, params = profile(wrap, inputs=(x,), verbose=False)
        else:
            dec_ctc_cfg = cfgs.get('decoder_side_ctc', {}) or {}
            wrap = _ProfileWrapper(
                model,
                kind='base',
                force_dec_ctc=bool(dec_ctc_cfg.get('enabled', False)),
            )
            macs, params = profile(wrap, inputs=(x,), verbose=False)

    results['macs'] = int(macs)
    results['params'] = int(params)
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
