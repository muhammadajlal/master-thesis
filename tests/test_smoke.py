"""CPU smoke tests: config placeholders resolve and the core models run forward.

Run with `pytest tests/` or `python tests/test_smoke.py`.
"""
import glob
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rewi.model import BaseModel  # noqa: E402
from rewi.utils import load_cfg, resolve_roots  # noqa: E402

NUM_CHAR = 61  # OnHW word categories incl. the CTC blank


def _walk_strings(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_strings(v)
    elif isinstance(obj, str):
        yield obj


def test_roots_default_inside_repo():
    roots = resolve_roots()
    assert os.path.samefile(roots['REPO'], ROOT)
    assert roots['DATA_ROOT'] and roots['RESULTS_ROOT']


def test_every_config_expands_without_placeholders():
    """Every config must resolve through the three placeholders only.

    DATA_ROOT / RESULTS_ROOT are pointed at sentinel directories so that any
    dataset or output path that bypasses the placeholders is detected.
    """
    paths = sorted(glob.glob(os.path.join(ROOT, 'configs', '**', '*.yaml'), recursive=True))
    assert paths, 'no configs found'
    saved = {k: os.environ.get(k) for k in ('REPO', 'DATA_ROOT', 'RESULTS_ROOT')}
    os.environ['REPO'] = ROOT
    os.environ['DATA_ROOT'] = '/__sentinel_data__'
    os.environ['RESULTS_ROOT'] = '/__sentinel_results__'
    bad = []
    try:
        for p in paths:
            cfg = load_cfg(p)
            assert isinstance(cfg, dict), p
            rel = os.path.relpath(p, ROOT)
            for s in _walk_strings(cfg):
                if '${' in s:
                    bad.append((rel, s))
            for key, root in (('dir_dataset', '/__sentinel_data__'), ('dir_work', '/__sentinel_results__')):
                v = cfg.get(key)
                if not isinstance(v, str) or not v:
                    continue  # empty values are filled in by the launch scripts
                if v.startswith('__') and v.endswith('__'):
                    continue  # template marker patched by scripts/repro/reproduce_tables.sh
                if not (v.startswith(root) or v.startswith(ROOT)):
                    bad.append((rel, f'{key}: {v}'))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert not bad, bad[:10]


def test_example_configs_have_required_keys():
    for p in glob.glob(os.path.join(ROOT, 'configs', 'examples', '*.yaml')):
        cfg = load_cfg(p)
        for key in ('arch_en', 'arch_de', 'dir_dataset', 'dir_work', 'epoch', 'categories'):
            assert key in cfg, f'{os.path.basename(p)} lacks {key}'
        assert cfg['idx_fold'] == -1, f'{os.path.basename(p)} should run all folds'


def test_ctc_model_forward():
    torch.manual_seed(0)
    model = BaseModel('blconv_b', 'bilstm_wide', in_chan=13, num_cls=NUM_CHAR).eval()
    x = torch.randn(2, 13, 512)
    with torch.no_grad():
        out = model(x)
    assert out.dim() == 3 and out.shape[0] == 2 and out.shape[-1] == NUM_CHAR, out.shape


def test_hwrformer_forward_teacher_forced():
    torch.manual_seed(0)
    vocab = NUM_CHAR + 3  # PAD, BOS, EOS
    model = BaseModel('blconv_b', 'ar_transformer_xs', in_chan=13, num_cls=vocab,
                      use_gated_attention=True, gating_type='elementwise').eval()
    x = torch.randn(2, 13, 512)
    y_inp = torch.randint(0, vocab, (2, 6))
    with torch.no_grad():
        logits = model(x, None, y_inp)
    assert logits.shape == (2, 6, vocab), logits.shape
    n_params = sum(p.numel() for p in model.parameters())
    assert 4.4e6 < n_params < 4.9e6, n_params  # parameter-matched to REWI (~4.64 M)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'ok  {name}')
