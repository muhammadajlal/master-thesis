import os
import random

import numpy as np
import torch
import yaml

# Directory roots that configs may reference as ${REPO}, ${DATA_ROOT} and
# ${RESULTS_ROOT}. Defaults keep everything inside the repository; override
# them in the environment to point at external dataset / result locations
# (e.g. on the thesis cluster: DATA_ROOT=$REPO/../../data,
# RESULTS_ROOT=$REPO/../../results).
_ROOT_DEFAULTS = {
    'REPO': lambda: os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'DATA_ROOT': lambda: os.path.join(os.environ['REPO'], 'data'),
    'RESULTS_ROOT': lambda: os.path.join(os.environ['REPO'], 'results'),
}


def resolve_roots() -> dict:
    '''Make sure REPO, DATA_ROOT and RESULTS_ROOT are set in the environment.

    Unset variables fall back to the repository-relative defaults above. The
    resolved values are returned as a dict.
    '''
    for key, default in _ROOT_DEFAULTS.items():
        if not os.environ.get(key):
            os.environ[key] = default()
    return {key: os.environ[key] for key in _ROOT_DEFAULTS}


def expand_cfg_paths(obj):
    '''Recursively expand ``${VAR}`` environment variables and ``~`` in every
    string of a loaded config, so configs can reference ``${REPO}/...``,
    ``${DATA_ROOT}/...`` and ``${RESULTS_ROOT}/...`` instead of absolute paths.

    Args:
        obj: Value loaded from YAML (dict, list or scalar).

    Returns:
        The value with all contained strings expanded.
    '''
    resolve_roots()
    if isinstance(obj, dict):
        return {k: expand_cfg_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_cfg_paths(v) for v in obj]
    if isinstance(obj, str) and ('$' in obj or obj.startswith('~')):
        return os.path.expanduser(os.path.expandvars(obj))
    return obj


def load_cfg(path: str) -> dict:
    '''Load a YAML config and expand the path placeholders in it.

    Args:
        path (str): Path to the YAML file.

    Returns:
        dict: Configuration with ``${REPO}``-style placeholders resolved.
    '''
    with open(path, 'r') as f:
        return expand_cfg_paths(yaml.safe_load(f))


def seed_everything(seed: int = 42) -> None:
    '''Seed everything in the environment.

    Args:
        seed (int, optional): Seed value. Defaults to 42.
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    '''Seed workers of dataloader.

    Args:
        worker_id (int): Worker ID.
    '''
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def sec2time(time_sec: float) -> str:
    '''Convert number of seconds to time in h:m:s format.

    Args:
        time (float): Number of seconds.

    Returns:
        str: String of time.
    '''
    second = str(int(time_sec % 60)).zfill(2)
    minute = str(int(time_sec // 60) % 60).zfill(2)
    hour = int(time_sec // 3600)
    time = f'{hour}:{minute}:{second}'

    return time
