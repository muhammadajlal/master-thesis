import argparse
import json
import os
from glob import glob

import numpy as np
import torch
import yaml
from thop import profile

from rewi.model import BaseModel, build_encoder
from rewi.model.multimodal_lm_model import MultimodalLMModel
from rewi.model.pretrainedLM import LMConfig


def _is_lm_mode(cfgs: dict) -> bool:
    arch_de = str(cfgs.get('arch_de', '')).lower()
    # We treat it as LM-mode only if a local HF model path/name is provided.
    return ('t5' in arch_de) and bool(cfgs.get('lm_name'))


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
    T = 1024 if 'word' in cfgs['dir_dataset'] else 4096

    # LM path (T5/ByT5 multimodal)
    if _is_lm_mode(cfgs):
        encoder = build_encoder(cfgs['num_channel'], cfgs['arch_en'], cfgs.get('len_seq', 0))
        ratio_ds = int(getattr(encoder, 'ratio_ds', 1))

        d_cnn = int(cfgs.get('d_cnn', getattr(encoder, 'dim_out')))
        lm_cfg = LMConfig(
            name=str(cfgs['lm_name']),
            train_lm=bool(cfgs.get('lm_train_lm', False)),
            max_new_tokens=int(cfgs.get('lm_max_new_tokens', 128)),
            num_beams=int(cfgs.get('lm_num_beams', 1)),
            length_penalty=float(cfgs.get('lm_length_penalty', 1.0)),
            min_new_tokens=int(cfgs.get('lm_min_new_tokens', 0)),
            local_files_only=True,
            no_repeat_ngram_size=int(cfgs.get('lm_no_repeat_ngram_size', 0)),
            repetition_penalty=float(cfgs.get('lm_repetition_penalty', 1.0)),
            early_stopping=bool(cfgs.get('lm_early_stopping', False)),
        )

        model = MultimodalLMModel(
            encoder=encoder,
            ratio_ds=ratio_ds,
            d_cnn=d_cnn,
            lm_cfg=lm_cfg,
            proj_dropout=float(cfgs.get('lm_proj_dropout', 0.0)),
            freeze_encoder=bool(cfgs.get('freeze', False)),
        ).eval()

        x = torch.randn(1, cfgs['num_channel'], T)
        len_x = torch.tensor([T], dtype=torch.long)

        # Provide a small dummy label sequence so forward() is runnable for profiling.
        vocab_size = int(model.lm.lm.config.vocab_size)
        labels = torch.randint(low=0, high=vocab_size, size=(1, 8), dtype=torch.long)

        try:
            macs, params = profile(model, inputs=(x, len_x, labels))
            results['macs'] = int(macs)
            results['params'] = int(params)
        except Exception as e:
            # HF models can be difficult for thop; keep evaluate.py usable anyway.
            results['params'] = int(sum(p.numel() for p in model.parameters()))
            results.setdefault('macs', -1)
            print(f"[WARN] thop profiling failed for LM model: {e}")

        results = {k: v for k, v in sorted(results.items())}
        return results

    # Non-LM path (original behavior)
    model = BaseModel(
        cfgs['arch_en'],
        cfgs['arch_de'],
        cfgs['num_channel'],
        len(cfgs['categories']),
        cfgs.get('len_seq', 0),
        use_gated_attention=bool(cfgs.get('use_gated_attention', False)),
        gating_type=str(cfgs.get('gating_type', 'elementwise')),
    ).eval()
    model.infer()
    x = torch.randn(1, cfgs['num_channel'], T)
    macs, params = profile(model, inputs=(x,))

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
