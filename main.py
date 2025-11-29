import argparse
import os
import warnings

import torch
import torch.nn as nn
import yaml
from loguru import logger
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from rewi.dataset import HRDataset
from rewi.dataset.utils import fn_collate
from rewi.evaluate import evaluate
from rewi.loss import CTCLoss
from rewi.manager import RunManager
from rewi.model import BaseModel
from rewi.utils import seed_everything, seed_worker
from rewi.visualize import visualize
from rewi.ctc_decoder import BestPath
from rewi.tokenizer import BPETokenizer

warnings.filterwarnings('ignore', category=UserWarning)

def build_ar_batch(y: torch.Tensor, len_y: torch.Tensor,
                   pad_id: int, bos_id: int, eos_id: int,
                   device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Supports:
      - y shape (B, S_max) with right-padding (common collate)
      - y shape (sum(len_y),) flat concat (rare)
    Returns:
      y_inp: [BOS, y1..yK] padded with PAD   -> (B, N)
      y_tgt: [y1..yK, EOS] padded with PAD   -> (B, N)
    """
    # Ensure CPU list of lengths
    lengths = len_y.tolist()
    parts = []

    if y.dim() == 2:
        # y is (B, S_max)
        B = y.size(0)
        for b in range(B):
            L = int(lengths[b])
            parts.append(y[b, :L])
    elif y.dim() == 1:
        # y is flat (sum(L),)
        parts = list(torch.split(y, tuple(int(l) for l in lengths), dim=0))
    else:
        raise ValueError(f"build_ar_batch: unexpected y.dim() = {y.dim()} (expected 1 or 2)")

    # Compute padded length (+1 for BOS/EOS shift)
    N = max(p.size(0) for p in parts) + 1
    B = len(parts)

    y_inp = torch.full((B, N), pad_id, dtype=torch.long, device=device)
    y_tgt = torch.full((B, N), pad_id, dtype=torch.long, device=device)

    for b, lab in enumerate(parts):
        L = lab.size(0)
        y_inp[b, 0] = bos_id
        if L > 0:
            y_inp[b, 1:L+1] = lab.to(torch.long)
            y_tgt[b, 0:L]   = lab.to(torch.long)
        y_tgt[b, L] = eos_id

    return y_inp, y_tgt







def train_one_epoch(
    dataloader: DataLoader,
    model: BaseModel,
    fn_loss: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    lr_scheduler: torch.optim.lr_scheduler.SequentialLR,
    man: RunManager,
    epoch: int,
) -> None:
    '''Train model for 1 epoch.

    Args:
        dataloader (torch.utils.data.DataLoader): Dataloader of training set.
        model (hwr.model.BaseModel): Model.
        fn_loss (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        scaler (torch.cuda.amp.GradScaler): Scaler for mix-precision training.
        lr_schedular (torch.optim.lr_scheduler.SequentialLR): Learning rate scheduler.
        man (hwr.manager.RunManager): Running manager.
        epoch (int): Current epoch number.
    '''
    man.initialize_epoch(epoch, len(dataloader), False)
    model.train()

    for idx, (x, y, len_x, len_y) in enumerate(dataloader):
        x, y = x.to(man.cfgs.device), y.to(man.cfgs.device)
        optimizer.zero_grad()
        """"
        with torch.autocast('cuda', torch.float16):
            out = model(x)
            loss = fn_loss(
                out.permute((1, 0, 2)), y, len_x // model.ratio_ds, len_y
            ) """
        
        # train_one_epoch
        PAD_ID, BOS_ID, EOS_ID = man.cfgs.PAD_ID, man.cfgs.BOS_ID, man.cfgs.EOS_ID

        with torch.autocast('cuda', torch.float16):
            if isinstance(fn_loss, nn.CrossEntropyLoss):  # AR mode
                # Build teacher-forcing inputs
                y_inp, y_tgt = build_ar_batch(
                    y, len_y, PAD_ID, BOS_ID, EOS_ID, device=man.cfgs.device
                )
                # Forward with masking (your BaseModel should handle in_lengths)
                logits = model(x, in_lengths=len_x, y_inp=y_inp)  # (B, N, V)
                loss = fn_loss(logits.reshape(-1, logits.size(-1)),
                            y_tgt.reshape(-1))
            else:
                # CTC path (unchanged)
                out = model(x)
                loss = fn_loss(
                    out.permute((1, 0, 2)), y, len_x // model.ratio_ds, len_y
                )


        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        man.update_iteration(
            idx,
            loss.item(),
            lr_scheduler.get_last_lr()[0],
        )

    man.summarize_epoch()

    # save checkpoints every freq_save epoch
    if man.check_step(epoch + 1, 'save'):
        man.save_checkpoint(
            model.state_dict(),
            optimizer.state_dict(),
            lr_scheduler.state_dict(),
        )


def test(
    dataloader: DataLoader,
    model: BaseModel,
    fn_loss: nn.Module,
    man: RunManager,
    ctc_decoder: BestPath,
    epoch: int | None = None,
    tokenizer=None,                 # ✅ add this
) -> None:
    '''Test the model.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader of test set.
        model (hwr.model.BaseModel): Model.
        fn_loss (torch.nn.Module): Loss function.
        man (hwr.manager.RunManager): Running manager.
        ctc_decoder (BestPath): An instance of CTC decoder.
        epoch (int | None, optional): Epoch number. Defaults to None.
    '''
    preds = []  # predictions for evaluation
    labels = []  # labels for evaluation
    man.initialize_epoch(epoch, len(dataloader), True)
    model.eval()
    
    PAD_ID, BOS_ID, EOS_ID = man.cfgs.PAD_ID, man.cfgs.BOS_ID, man.cfgs.EOS_ID
    with torch.no_grad():
        for idx, (x, y, len_x, len_y) in enumerate(dataloader):
            x, y = x.to(man.cfgs.device), y.to(man.cfgs.device)

            if isinstance(fn_loss, nn.CrossEntropyLoss):
            # AR path
                y_inp, y_tgt = build_ar_batch(y, len_y, PAD_ID, BOS_ID, EOS_ID, device=man.cfgs.device)
                logits = model(x, in_lengths=len_x, y_inp=y_inp)
                loss = fn_loss(logits.reshape(-1, logits.size(-1)), y_tgt.reshape(-1))
            else:
                # CTC path
                out = model(x)
                loss = fn_loss(out.permute((1,0,2)), y, len_x // model.ratio_ds, len_y)

            man.update_iteration(idx, loss.item())

        # Only CTC has a decoder implemented here
            if man.check_step(epoch + 1, 'eval') and not isinstance(fn_loss, nn.CrossEntropyLoss):
                for pred, len_pred, label in zip(out.cpu(), len_x // model.ratio_ds, y.cpu()):
                    preds.append(ctc_decoder.decode(pred[:len_pred]))
                    labels.append(ctc_decoder.decode(label, True))

            # >>> ADD THIS: AR greedy decoding for logging <<<

            if man.check_step(epoch + 1, 'eval') and isinstance(fn_loss, nn.CrossEntropyLoss):
                B = x.size(0)
                max_len = int(len_y.max().item()) + 2
                device = x.device

                #tok = getattr(man.cfgs, 'tokenizer_obj', None)     # BPE tokenizer or None
                tok = tokenizer   # ✅ use the passed-in object
                chars = getattr(man.cfgs, 'categories', None)      # fallback for char mode

                # autoregressively grow y from BOS
                y_gen = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
                for _ in range(max_len):
                    step_logits = model(x, in_lengths=len_x, y_inp=y_gen)   # (B, t, V)
                    nxt = step_logits[:, -1, :].argmax(-1, keepdim=True)    # (B,1)
                    y_gen = torch.cat([y_gen, nxt], dim=1)                  # (B, t+1)

                y_gen = y_gen.detach().cpu().tolist()
                y_cpu = y.cpu()
                len_y_cpu = len_y.cpu().tolist()

                for b in range(B):
                    # --- prediction ---
                    ids_pred = y_gen[b][1:]  # skip BOS
                    seq = []
                    for t in ids_pred:
                        if t == EOS_ID: break
                        if t == PAD_ID: continue
                        seq.append(int(t))

                    if tok is not None:
                        # BPE path
                        pred_str = tok.decode(seq)
                    else:
                        # char fallback (skip CTC blank=0 if present in your set)
                        pred_str = ''.join(
                            chars[i] for i in seq
                            if (chars is not None) and 0 <= i < len(chars) and i != 0
                        )
                    preds.append(pred_str)

                    # --- label (ground truth) ---
                    if y.dim() == 2:
                        L = int(len_y_cpu[b])
                        lab_ids = y_cpu[b, :L].tolist()
                        if tok is not None:
                            lab_str = tok.decode(lab_ids)
                        else:
                            lab_str = ''.join(
                                chars[i] for i in lab_ids
                                if (chars is not None) and 0 <= i < len(chars) and i != 0
                            )
                        labels.append(lab_str)
       


    man.summarize_epoch()

    # Evaluation and visualization
    if man.check_step(epoch + 1, 'eval'):
        if not isinstance(fn_loss, nn.CrossEntropyLoss):
            visualize(preds, labels, man.cfgs.categories[1:], man.dir_vis, epoch)
            results_eval = evaluate(preds, labels)
            man.update_evaluation(results_eval, preds[:20], labels[:20])
        else:
            # AR mode: optionally log perplexity or average CE instead
            # >>> ADD THIS: evaluate AR predictions <<<
            results_eval = evaluate(preds, labels)
            man.update_evaluation(results_eval, preds[:20], labels[:20])



def main(cfgs: argparse.Namespace) -> None:
    '''Main function for training and evaluation.

    Args:
        cfgs (argparse.Namespace): Configurations.
    '''

    # 1) AR mode
    AR_MODE = cfgs.arch_de in {"ar_transformer_s", "ar_transformer_m", "ar_transformer_l"}

    # 2) Tokenizer setup (BPE optional) — define tok BEFORE using it
    tok = None
    if getattr(cfgs, 'use_bpe', False):
        cfgs.tokenizer_model_path = cfgs.tokenizer['model']  # ✅ string is serializable


    # 3) Compute vocab + special IDs ONCE (no duplicate overrides below!)
    if tok is not None:
        vocab_dec = tok.vocab_size           # e.g., 100
        PAD_ID, BOS_ID, EOS_ID = tok.PAD, tok.BOS, tok.EOS
    else:
        # char mode: categories + 3 specials
        base = len(cfgs.categories)
        PAD_ID, BOS_ID, EOS_ID = base, base + 1, base + 2
        vocab_dec = base + (3 if AR_MODE else 0)

    # 4) Attach to cfgs BEFORE constructing RunManager / datasets
    cfgs.AR_MODE = AR_MODE
    cfgs.PAD_ID, cfgs.BOS_ID, cfgs.EOS_ID = PAD_ID, BOS_ID, EOS_ID
    cfgs.vocab_dec = vocab_dec
    cfgs.tokenizer_obj = tok  # now tok definitely exists

    # 5) Proceed as usual
    manager = RunManager(cfgs)
    seed_everything(cfgs.seed)
    ctc_decoder = BestPath(cfgs.categories)

    model = BaseModel(
        cfgs.arch_en,
        cfgs.arch_de,
        cfgs.num_channel,
        cfgs.vocab_dec,      # use vocab_dec from cfgs
        cfgs.len_seq,
    ).to(cfgs.device)

    # Datasets: hand tokenizer to datasets
    dataset_test = HRDataset(
        os.path.join(cfgs.dir_dataset, 'val.json'),
        cfgs.categories,
        model.ratio_ds,
        cfgs.idx_fold,
        cfgs.len_seq,
        cache=cfgs.cache,
    )
    dataset_test.tokenizer = tok

    dataloader_test = DataLoader(
        dataset_test, cfgs.size_batch, num_workers=cfgs.num_worker, collate_fn=fn_collate,
    )

    fn_loss = (nn.CrossEntropyLoss(ignore_index=cfgs.PAD_ID, label_smoothing=0.1)
               if AR_MODE else CTCLoss())

    epoch_start = 0

    if not cfgs.test:
        dataset_train = HRDataset(
            os.path.join(cfgs.dir_dataset, 'train.json'),
            cfgs.categories,
            model.ratio_ds,
            cfgs.idx_fold,
            cfgs.len_seq,
            cfgs.aug,
            cfgs.cache,
        )
        dataset_train.tokenizer = tok

        dataloader_train = DataLoader(
            dataset_train,
            cfgs.size_batch,
            True,
            num_workers=cfgs.num_worker,
            collate_fn=fn_collate,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(cfgs.seed),
        )

        optimizer = torch.optim.AdamW(model.parameters(), cfgs.lr)
        scaler = GradScaler()
        lr_scheduler = SequentialLR(
            optimizer,
            [
                LinearLR(optimizer, 0.01, total_iters=len(dataloader_train) * cfgs.epoch_warmup),
                CosineAnnealingLR(optimizer, len(dataloader_train) * (cfgs.epoch - cfgs.epoch_warmup)),
            ],
            [len(dataloader_train) * cfgs.epoch_warmup],
        )
    # ... keep the rest of your code (checkpoint load, training loop, etc.) unchanged


    # load checkpoint if given
    if cfgs.checkpoint:
        ckp = torch.load(cfgs.checkpoint, weights_only=False)
        model.load_state_dict(ckp['model'], strict=False)

        if not cfgs.test:
            if 'epoch' in ckp.keys():  # resume
                epoch_start = ckp['epoch'] + 1
                optimizer.load_state_dict(ckp['optimizer'])
                lr_scheduler.load_state_dict(ckp['lr_scheduler'])
            elif cfgs.freeze:  # freeze
                for params in model.encoder.parameters():
                    params.requires_grad = False
            else:  # finetune
                optimizer = torch.optim.AdamW(
                    [
                        {
                            'params': model.encoder.parameters(),
                            'lr': cfgs.lr * 0.1,
                        },
                        {
                            'params': model.decoder.parameters(),
                            'lr': cfgs.lr,
                        },
                    ]
                )
                lr_scheduler = SequentialLR(
                    optimizer,
                    [
                        LinearLR(
                            optimizer,
                            0.01,
                            total_iters=len(dataloader_train)
                            * cfgs.epoch_warmup,
                        ),
                        CosineAnnealingLR(
                            optimizer,
                            len(dataloader_train)
                            * (cfgs.epoch - cfgs.epoch_warmup),
                        ),
                    ],
                    [len(dataloader_train) * cfgs.epoch_warmup],
                )

        logger.info(f'Load checkpoint from {cfgs.checkpoint}')

    # start running
    for e in range(epoch_start, cfgs.epoch):
        if cfgs.test:
            test(
                dataloader_test,
                model,
                fn_loss,
                manager,
                ctc_decoder,
                -1,
                tokenizer=tok          # ✅ pass tokenizer
            )
            break
        else:
            train_one_epoch(
                dataloader_train,
                model,
                fn_loss,
                optimizer,
                scaler,
                lr_scheduler,
                manager,
                e,
            )
            test(
                dataloader_test,
                model,
                fn_loss,
                manager,
                ctc_decoder,
                e,
                tokenizer=tok          # ✅ pass tokenizer
            )

    if not cfgs.test:
        manager.summarize_evaluation()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run handwriting recognition model.'
    )
    parser.add_argument(
        '-c', '--config', help='Path to YAML file of configuration.'
    )
    args = parser.parse_args()
    # args.config = 'configs/train.yaml'

    with open(args.config, 'r') as f:
        cfgs = yaml.safe_load(f)
        cfgs = argparse.Namespace(**cfgs)

    main(cfgs)
