"""Run k-fold cross-validation training sequentially.

Generates one config per fold from a base config with ``idx_fold: -1``, runs
``main.py`` on each, and removes the temporary configs afterwards. Path
placeholders (``${REPO}``, ``${DATA_ROOT}``, ``${RESULTS_ROOT}``) are resolved
before the per-fold configs are written.

Usage:
    python train_cv.py -c configs/examples/hwrformer_onhw_wi.yaml
"""
import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rewi.utils import load_cfg  # noqa: E402


def train_cv(cfgs: dict, path_main: str) -> None:
    '''Train model in a cross validation fashion.

    Args:
        cfgs (dict): Training configuration.
        path_main (str): Path to the Python script for training.
    '''
    with open(os.path.join(cfgs['dir_dataset'], 'train.json'), 'r') as f:
        num_fd = json.load(f)['info']['num_fold']

    dir_temp = f'temp_{os.path.basename(cfgs["dir_work"])}'

    os.makedirs(dir_temp, exist_ok=True)

    command = []
    seperator = ' && '

    for i in range(num_fd):
        cfgs['idx_fold'] = i
        path_temp = os.path.join(dir_temp, f'f{i}.yaml')

        with open(path_temp, 'w') as f:
            yaml.safe_dump(cfgs, f)

        command.append(f'{sys.executable} {path_main} -c {path_temp}')

    command = seperator.join(command) + f' && rm -rf {dir_temp}'
    print(command)
    os.system(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run handwriting recognition model with cross validation.'
    )
    parser.add_argument(
        '-c', '--config', required=True, help='Path to the YAML file of configuration.'
    )
    parser.add_argument(
        '-m',
        '--main',
        help='Path to the Python script for training.',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main.py'),
    )
    args = parser.parse_args()

    cfgs = load_cfg(args.config)

    assert (
        cfgs['idx_fold'] == -1
    ), 'Please use cross-validation training configuration.'

    train_cv(cfgs, args.main)
