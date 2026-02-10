import argparse
import json
import os
import time
from datetime import datetime
from typing import Any

import torch
import yaml
from loguru import logger

from .utils import sec2time


class RunManager:
    '''Manage work directory, logging, losses, evaluation results and
    checkpoints.
    '''

    def __init__(self, cfgs: argparse.Namespace) -> None:
        '''Manage work directory, logging, losses, evaluation results and
        checkpoints.

        Args:
            cfgs (argparse.Namespace): Configurations.
        '''
        self.cfgs = cfgs
        self.results = {}
        self.ts = datetime.now().strftime('%Y%m%d%H%M%S')

        self.initialize_directory()

    def check_step(self, scur: int, mode: str) -> bool:
        '''Check whether the current step is desired according to the given
        frequency. The last step is always True regardless the frequency. CALL
        AFTER initialize_epoch!

        Args:
            scur (int): Current step number.
            model (str): Type of the step. Options are 'iter', 'eval', and 'save'.

        Returns:
            bool: Whether the current step is desired.
        '''
        match mode:
            case 'iter':
                return scur % self.cfgs.freq_log == 0 or scur == self.num_iter
            case 'eval':
                return (
                    scur % self.cfgs.freq_eval == 0 or scur == self.cfgs.epoch
                )
            case 'save':
                return (
                    scur % self.cfgs.freq_save == 0 or scur == self.cfgs.epoch
                )

    def initialize_directory(self) -> None:
        '''Initialize the work directory.'''
        tag = 'test' if self.cfgs.test else 'train'

        self.dir_ckp = os.path.join(
            self.cfgs.dir_work, str(self.cfgs.idx_fold), 'checkpoints'
        )
        self.dir_vis = os.path.join(
            self.cfgs.dir_work, str(self.cfgs.idx_fold), 'visualization'
        )
        path_cfg = os.path.join(
            self.cfgs.dir_work,
            str(self.cfgs.idx_fold),
            f'{tag}_{self.ts}.yaml',
        )
        path_log = os.path.join(
            self.cfgs.dir_work, str(self.cfgs.idx_fold), f'{tag}_{self.ts}.log'
        )
        self.path_result = os.path.join(
            self.cfgs.dir_work,
            str(self.cfgs.idx_fold),
            f'{tag}_{self.ts}.json',
        )

        os.makedirs(self.dir_ckp, exist_ok=True)
        os.makedirs(self.dir_vis, exist_ok=True)

        with open(os.path.join(path_cfg), 'w') as f:
            yaml.safe_dump(vars(self.cfgs), f)

        logger.add(path_log)
        logger.info(
            f'Initialized work directory at {self.cfgs.dir_work} '
            f'for fold {self.cfgs.idx_fold}.'
        )

        # Debug-friendly run summary.
        # Prefer derived flags set in main.py, but fall back to local inference
        # so other entrypoints/scripts still print the regime.
        regime = getattr(self.cfgs, "TRAINING_REGIME", None)

        dual_cfg = getattr(self.cfgs, "dual_head", None)
        if isinstance(dual_cfg, dict):
            dual_enabled = bool(dual_cfg.get("enabled", False))
        else:
            dual_enabled = bool(getattr(dual_cfg, "enabled", False))

        if regime is None:
            arch_de = str(getattr(self.cfgs, "arch_de", ""))
            if dual_enabled:
                regime = "hybrid_ar_ctc"
            elif arch_de.startswith("ar_"):
                regime = "ar_only"
            else:
                regime = "ctc_only"

        logger.info(
            "[RunSummary] regime={} | test={} | fold={} | device={} | arch_en={} | arch_de={} | use_bpe={} | size_batch={} | lr={} | seed={}",
            str(regime),
            bool(getattr(self.cfgs, "test", False)),
            int(getattr(self.cfgs, "idx_fold", -1)),
            str(getattr(self.cfgs, "device", "?")),
            str(getattr(self.cfgs, "arch_en", "?")),
            str(getattr(self.cfgs, "arch_de", "?")),
            bool(getattr(self.cfgs, "use_bpe", False)),
            int(getattr(self.cfgs, "size_batch", -1)),
            float(getattr(self.cfgs, "lr", -1.0)),
            int(getattr(self.cfgs, "seed", -1)),
        )

        if bool(getattr(self.cfgs, "DUAL_HEAD", False)) or dual_enabled:
            dual = dual_cfg or {}
            logger.info(
                "[RunSummary][DualHead] arch_ctc={} | lambda_ar={} | lambda_ctc={} | vocab_ar={} | vocab_ctc={}",
                str(dual.get("arch_ctc", getattr(self.cfgs, "dual_head_arch_ctc", "?"))),
                float(dual.get("lambda_ar", 1.0)),
                float(dual.get("lambda_ctc", 0.0)),
                int(getattr(self.cfgs, "vocab_dec", -1)),
                int(getattr(self.cfgs, "vocab_ctc", -1)),
            )
            # Log schedule and balance mode if set
            sched = dual.get("lambda_ctc_schedule", None)
            balance = dual.get("loss_balance", "sum")
            if sched:
                logger.info(
                    "[RunSummary][DualHead][Schedule] type={} | warmup_epochs={} | max={} | decay_start={} | min={} | balance={}",
                    str(sched.get("type", "linear")),
                    int(sched.get("warmup_epochs", 0)),
                    float(sched.get("max", dual.get("lambda_ctc", 0.3))),
                    sched.get("decay_start", None),
                    float(sched.get("min", sched.get("max", dual.get("lambda_ctc", 0.3)))),
                    str(balance),
                )
            else:
                logger.info("[RunSummary][DualHead][Schedule] none (static lambda_ctc) | balance={}", str(balance))

        # Persist run metadata inside the results JSON so aggregation scripts can
        # understand the training regime/hyperparameters without parsing logs.
        meta: dict[str, Any] = {
            "ts": str(self.ts),
            "tag": str(tag),
            "dir_work": str(getattr(self.cfgs, "dir_work", "")),
            "idx_fold": int(getattr(self.cfgs, "idx_fold", -1)),
            "test": bool(getattr(self.cfgs, "test", False)),
            "training_regime": str(regime),
            "arch_en": str(getattr(self.cfgs, "arch_en", "?")),
            "arch_de": str(getattr(self.cfgs, "arch_de", "?")),
            "use_bpe": bool(getattr(self.cfgs, "use_bpe", False)),
            "size_batch": int(getattr(self.cfgs, "size_batch", -1)),
            "lr": float(getattr(self.cfgs, "lr", -1.0)),
            "seed": int(getattr(self.cfgs, "seed", -1)),
        }

        if bool(getattr(self.cfgs, "DUAL_HEAD", False)) or dual_enabled:
            dual = dual_cfg or {}
            meta["dual_head"] = {
                "enabled": True,
                "arch_ctc": str(dual.get("arch_ctc", getattr(self.cfgs, "dual_head_arch_ctc", "?"))),
                "lambda_ar": float(dual.get("lambda_ar", 1.0)),
                "lambda_ctc": float(dual.get("lambda_ctc", 0.0)),
                "vocab_ar": int(getattr(self.cfgs, "vocab_dec", -1)),
                "vocab_ctc": int(getattr(self.cfgs, "vocab_ctc", -1)),
                "loss_balance": str(dual.get("loss_balance", "sum")),
                "lambda_ctc_schedule": dual.get("lambda_ctc_schedule", None),
            }
        else:
            meta["dual_head"] = {"enabled": False}

        self.results.setdefault("meta", meta)
        # Write an initial JSON right away (useful for debugging & for early-crash runs)
        self.save_results()

    def initialize_epoch(self, epoch: int, num_iter: int, val: bool) -> None:
        '''Initialize the recording for new epoch.

        Args:
            epoch (int): Epoch number.
            num_iter (int): Maximum number of iterations of the current epoch.
            val (bool): Whether the current epoch is for training phase, or validation or test phases.
        '''
        self.epoch = epoch  # epoch life time
        self.loss = []  # epoch life time
        self.loss_components: dict[str, list[float]] = {}  # epoch life time
        self.num_iter = num_iter  # epoch life time
        self.t_start = time.time()  # epoch life time
        self.tag = 'test' if val else 'train'  # epoch life time

        if not epoch in self.results.keys():
            self.results[epoch] = {}  # epoch life time

        self.results[epoch][self.tag] = []  # epoch life time

    def log(self, message: str) -> None:
        '''Log messages.

        Args:
            message (str): Message to log.
        '''
        logger.info(message)

    def save_checkpoint(
        self,
        state_model: dict | None = None,
        state_optimizer: dict | None = None,
        state_lr_scheduler: dict | None = None,
        filename: str | None = None,
    ) -> None:
        '''Save the states of model, optimizer and scheduler to the checkpoint
        directory in the work directory.

        Args:
            state_model (dict, optional): State dictionary of the model. Defaults to None.
            state_optimizer (dict, optional): State dictionary of the optimizer. Defaults to None.
            state_lr_scheduler (dict, optional): State dictionary of the learning rate scheduler. Defaults to None.
            filename (str | None, optional): Override output filename (e.g., "best_cer.pth"). Defaults to None.
        '''
        if filename is None:
            filename = f'{self.epoch}.pth'
        torch.save(
            {
                'epoch': self.epoch,
                'lr_scheduler': state_lr_scheduler,
                'model': state_model,
                'optimizer': state_optimizer,
            },
            os.path.join(self.dir_ckp, filename),
        )
        logger.info(f'Saved checkpoint: {filename} (epoch {self.epoch})')

    def save_results(self) -> None:
        '''Save the cached result dictionary as a JSON file to the work
        directory.
        '''
        with open(self.path_result, 'w') as f:
            json.dump(self.results, f)

    def summarize_epoch(self) -> float:
        '''Summarize and save the results of the epoch. CALL AFTER
        initialize_epoch!

        Returns:
            float: Average loss of the epoch.
        '''
        t_end = time.time() - self.t_start
        loss_avg = sum(self.loss) / len(self.loss)
        result: dict[str, Any] = {'loss_avg': loss_avg, 'time': t_end}
        for k, vs in self.loss_components.items():
            if vs:
                result[f'{k}_avg'] = float(sum(vs) / len(vs))
        logger.info(
            (
                f'{self.tag}, epoch: {self.epoch}, loss avg: {loss_avg:.7f}, '
                f'time: {sec2time(t_end)}'
            )
        )

        if self.loss_components:
            extra_msg = ", ".join(
                f"{k}_avg: {float(sum(vs) / len(vs)):.7f}"
                for k, vs in self.loss_components.items()
                if vs
            )
            if extra_msg:
                logger.info(f"{self.tag}, epoch: {self.epoch}, {extra_msg}")
        self.results[self.epoch][self.tag].append(result)
        self.save_results()

        return loss_avg

    def summarize_evaluation(self, key: str = 'evaluation') -> None:
        """Summarize evaluation results for a given key.

        By default this behaves exactly like the previous implementation and
        stores the best metrics under self.results['best'].

        For alternate evaluation keys (e.g. 'evaluation_ctc'), this stores the
        best metrics under self.results[f'best_{key}'].
        """
        results_eval = [
            [epoch, result[key]]
            for epoch, result in self.results.items()
            if isinstance(epoch, int) and isinstance(result, dict) and key in result.keys()
        ]
        if not results_eval:
            # Can happen early in training if freq_eval skips first epochs.
            return
        metrics = results_eval[0][1].keys()
        best = {metric: [-1, -1] for metric in metrics}  # [epoch, value]

        # iterate all results to get the best of each metrics
        for result in results_eval:
            for metric in metrics:
                if (
                    result[1][metric] < best[metric][1]
                    or best[metric][0] == -1
                ):
                    best[metric] = [result[0], float(result[1][metric])]

        store_key = 'best' if key == 'evaluation' else f'best_{key}'
        self.results[store_key] = best
        logger.info(f'best[{store_key}]: {best}')
        self.save_results()

    def update_evaluation(
        self,
        result: dict,
        preds: list[str] | None = None,
        labels: list[str] | None = None,
        key: str = 'evaluation',
        label: str | None = None,
    ) -> None:
        '''Update the evaluation results. Log predictions and labels if they
        are given.

        Args:
            result (dict): Evaluation results.
            preds (list[str] | None, optional): Predictions. Defaults to None.
            labels (list[str] | None, optional): Labels. Defaults to None.
        '''
        self.results[self.epoch][key] = result
        self.save_results()
        label_str = label if label is not None else key
        msg_log = [f'{metric}: {val:.7f}' for metric, val in result.items()]
        logger.info(f'[Eval][{label_str}][{key}] ' + ', '.join(msg_log))

        if preds:
            logger.info(f'[Eval][{label_str}][{key}] predictions: {preds}')

        if labels:
            logger.info(f'[Eval][{label_str}][{key}] labels: {labels}')

    def update_iteration(
        self,
        iter: int,
        loss: float,
        lr: float = -1,
        **extras: float,
    ) -> None:
        '''Update the status of the iteration. If the current iteration is
        desired according to the given frequency, log the information. CALL
        AFTER initialize_epoch!

        Args:
            iter (int): Iteration number.
            loss (float): Loss value.
            lr (float, optional): Current learning rate. Defaults to -1.
        '''
        self.loss.append(loss)
        for k, v in extras.items():
            try:
                fv = float(v)
            except Exception:
                continue
            self.loss_components.setdefault(k, []).append(fv)

        if self.check_step(iter + 1, 'iter'):
            t_inter = time.time() - self.t_start
            result = {
                'lr': lr,
                'iters': iter + 1,
                'loss': loss,
                'time': t_inter,
            }
            for k, v in extras.items():
                try:
                    result[k] = float(v)
                except Exception:
                    pass
            self.results[self.epoch][self.tag].append(result)

            extra_str = ""
            if extras:
                parts = []
                for k, v in extras.items():
                    try:
                        parts.append(f"{k}: {float(v):.7f}")
                    except Exception:
                        continue
                if parts:
                    extra_str = ", " + ", ".join(parts)

            logger.info(
                (
                    f'{self.tag}, epoch: {self.epoch}, iters: {iter + 1}/'
                    f'{self.num_iter}, lr: {lr:.7f}, loss: {loss:.7f}'
                    f'{extra_str}, time: {sec2time(t_inter)}'
                )
            )
