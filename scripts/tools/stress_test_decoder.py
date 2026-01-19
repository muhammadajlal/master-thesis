# stress_test_decoder.py — capacity + (optional) quality on real concatenations
import os, argparse, time, traceback, csv, re, glob
import torch, yaml
from types import SimpleNamespace

from rewi.model import BaseModel
from rewi.dataset import HRDataset
from rewi.evaluate import evaluate
from rewi.tokenizer import BPETokenizer  # used iff use_bpe: true

# -------- set your checkpoint here --------
CKPT_PATH = "/home/woody/iwso/iwso214h/imu-hwr/results/hwr2/blconv_ARDecoder_no_tokenizer/ar_transformer_s__wi_sent_hw6_meta/fold_3/3/checkpoints/254.pth"
# -----------------------------------------

def pick_device(cli_device: str | None, yaml_device: str | None) -> str:
    if cli_device in ("cpu", "cuda"):
        if cli_device == "cuda" and not torch.cuda.is_available():
            print("[WARN] --device cuda requested but CUDA is not available. Falling back to CPU.")
            return "cpu"
        return cli_device
    if yaml_device == "cuda" and not torch.cuda.is_available():
        print("[WARN] YAML device=cuda but CUDA is not available. Falling back to CPU.")
        return "cpu"
    if yaml_device in ("cpu", "cuda"):
        return yaml_device
    return "cuda" if torch.cuda.is_available() else "cpu"

def build_model(cfg, device):
    ar_mode = ("ar_transformer" in cfg.arch_de) or getattr(cfg, "AR_MODE", False)
    if getattr(cfg, "use_bpe", False) and hasattr(cfg, "tokenizer"):
        PAD_ID = getattr(cfg, "PAD_ID", 0)
        BOS_ID = getattr(cfg, "BOS_ID", 1)
        EOS_ID = getattr(cfg, "EOS_ID", 2)
        vocab_dec = cfg.tokenizer.get("vocab_size", 100)
    else:
        nchar = len(cfg.categories)
        PAD_ID = nchar
        BOS_ID = nchar + 1
        EOS_ID = nchar + 2
        vocab_dec = nchar + (3 if ar_mode else 0)
    cfg.PAD_ID, cfg.BOS_ID, cfg.EOS_ID = PAD_ID, BOS_ID, EOS_ID

    model = BaseModel(cfg.arch_en, cfg.arch_de, cfg.num_channel, vocab_dec, cfg.len_seq).to(device)
    model.eval()
    return model, PAD_ID, BOS_ID, EOS_ID

@torch.no_grad()
def ar_greedy_steps(model, x, len_x, BOS_ID, steps=12, amp=False):
    device = x.device
    y = torch.full((x.size(0), 1), BOS_ID, dtype=torch.long, device=device)
    use_amp = (amp and device.type == "cuda")
    for _ in range(steps):
        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(x, in_lengths=len_x, y_inp=y)
        else:
            logits = model(x, in_lengths=len_x, y_inp=y)
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        y = torch.cat([y, nxt], dim=1)
    return y

@torch.no_grad()
def ar_greedy_eos(model, x, len_x, BOS_ID, EOS_ID, max_steps, amp=False):
    device = x.device
    B = x.size(0)
    y = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    use_amp = (amp and device.type == "cuda")
    for _ in range(max_steps):
        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(x, in_lengths=len_x, y_inp=y)
        else:
            logits = model(x, in_lengths=len_x, y_inp=y)
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        y = torch.cat([y, nxt], dim=1)
        finished |= (nxt.squeeze(1) == EOS_ID)
        if torch.all(finished):
            break
    return y

def try_length(model, device, C, T, BOS_ID, steps, B=2, amp=False):
    x = torch.randn(B, C, T, device=device)
    len_x = torch.full((B,), T, dtype=torch.long, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    _ = ar_greedy_steps(model, x, len_x, BOS_ID, steps=steps, amp=amp)
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated()
    else:
        peak_mem = 0
    dt = time.time() - t0
    return dt, peak_mem

# ============== helpers for real concatenations ==============

def _trim_at_pad(y: torch.Tensor, pad_id: int | None):
    if pad_id is None:
        return y
    # keep tokens up to first PAD (exclusive); if no PAD, keep all
    idx = (y == pad_id).nonzero(as_tuple=True)[0]
    return y[: idx[0]] if idx.numel() else y

def _infer_len_x_from_signal(xi: torch.Tensor) -> int:
    xi = torch.as_tensor(xi)
    # assume (C,T) or (T,C); T is the larger dim typically
    if xi.ndim != 2:
        raise ValueError(f"Expected 2D (C,T)/(T,C), got {tuple(xi.shape)}")
    return max(xi.shape)

def _safe_get_item(dataset, idx, pad_id: int | None = None):
    """
    Works with datasets that return (x, y) OR (x, y, len_x, len_y).
    - Trims y to len_y if provided.
    - Optionally strips everything after first PAD and removes PADs.
    Returns: xi (Tensor), yi (Tensor), li_x (int), li_y (int)
    """
    sample = dataset[idx]

    if isinstance(sample, (list, tuple)) and len(sample) == 4:
        xi, yi, li_x, li_y = sample
        li_x = int(li_x)
        li_y = int(li_y)
        yi = torch.as_tensor(yi)[:li_y]
        yi = torch.as_tensor(_trim_at_pad(yi, pad_id))
        li_y = int(yi.numel())

    elif isinstance(sample, (list, tuple)) and len(sample) == 2:
        xi, yi = sample
        yi = torch.as_tensor(yi)
        yi = torch.as_tensor(_trim_at_pad(yi, pad_id))
        li_y = int(yi.numel())
        li_x = _infer_len_x_from_signal(xi)

    else:
        raise ValueError(f"Unexpected dataset item format from index {idx}: {type(sample)} len={len(sample) if isinstance(sample,(list,tuple)) else 'n/a'}")

    xi = torch.as_tensor(xi)
    yi = torch.as_tensor(yi)
    return xi, yi, li_x, li_y


def _to_C_T(xi: torch.Tensor) -> torch.Tensor:
    """
    Ensure per-sample tensor is (C, T).
    If it looks like (T, C) (i.e., first dim >> second), transpose it.
    """
    assert xi.dim() == 2, f"Expected 2D sample, got shape {tuple(xi.shape)}"
    C_like, T_like = xi.size(0), xi.size(1)
    # Heuristic: channels are small (<=64), time is large (hundreds/thousands)
    if C_like > T_like and T_like <= 64:
        xi = xi.transpose(0, 1)  # (T, C) -> (C, T)
    return xi

def make_concat_batch(dataset, B, items_per_concat, device, target_T=None, pad_id=None):
    """
    Build B samples by concatenating items end-to-end along time.
    Returns:
      x_batch: (B, C, T_max) float tensor (padded right, on `device`)
      len_x:   (B,) true lengths (on `device`)
      y_list:  list[list[int]] concatenated label id sequences
    """
    x_list, len_list, y_list = [], [], []
    idx = 0
    N = len(dataset)

    # Peek first item to infer C and dtype
    xi0, yi0, li_x0, li_y0 = _safe_get_item(dataset, idx, pad_id=pad_id)
    xi0 = _to_C_T(xi0)
    C = xi0.size(0)
    base_dtype = xi0.dtype

    for b in range(B):
        xs, ys, tot_T = [], [], 0
        tries = 0

        while True:
            # ⚠️ pass pad_id here
            xi, yi, li_x, li_y = _safe_get_item(dataset, idx, pad_id=pad_id)
            xi = _to_C_T(xi)

            # Handle rare channel mismatches robustly
            if xi.size(0) != C:
                if xi.transpose(0, 1).size(0) == C:
                    xi = xi.transpose(0, 1)
                else:
                    idx = (idx + 1) % N
                    tries += 1
                    if tries > items_per_concat * 40:
                        break
                    continue

            # strip padded tail
            xi = xi[:, :li_x]
            xs.append(xi)
            ys.extend(yi[:li_y].tolist())
            tot_T += li_x

            idx = (idx + 1) % N
            tries += 1

            if target_T is not None:
                # stop once we reach or slightly exceed target length
                if tot_T >= target_T:
                    break
                if tries >= items_per_concat * 20:  # safety limit for tiny items
                    break
            else:
                if tries >= items_per_concat:
                    break

        if not xs:
            x_cat = torch.zeros(C, 1, dtype=base_dtype)  # empty fallback
            ys = []
        else:
            x_cat = torch.cat(xs, dim=1)

        if target_T is not None and x_cat.size(1) > target_T:
            x_cat = x_cat[:, :target_T]

        x_list.append(x_cat)
        len_list.append(x_cat.size(1))
        y_list.append(ys)

    T_max = max(len_list) if len_list else 1

    # allocate directly on target device
    x_batch = torch.zeros(B, C, T_max, dtype=base_dtype, device=device)
    for b, xb in enumerate(x_list):
        T = xb.size(1)
        x_batch[b, :, :T] = xb.to(device=device)

    len_x = torch.tensor(len_list, dtype=torch.long, device=device)
    return x_batch, len_x, y_list



def ids_to_text(seq_ids, PAD_ID, BOS_ID, EOS_ID, chars=None, tok=None):
    # seq_ids: list[int]
    out_ids = []
    for t in seq_ids:
        if t == EOS_ID: break
        if t == PAD_ID or t == BOS_ID: continue
        out_ids.append(int(t))
    if tok is not None:
        return tok.decode(out_ids)
    if chars is not None:
        # skip CTC blank at 0 if present
        return ''.join(chars[i] for i in out_ids if 0 <= i < len(chars) and i != 0)
    return ''

# ============================== main ==============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="Path to YAML")
    ap.add_argument("--lengths", default="1024,2048,4096,8192,12288,16384",
                    help="Comma-separated input lengths T to try")
    ap.add_argument("--batches", default="8,16,32,64",
                    help="Comma-separated batch sizes B to try")
    ap.add_argument("--steps_list", default="8,16,32,64,128",
                    help="Comma-separated decode steps to try per (T,B)")
    ap.add_argument("--device", default="cuda", choices=["cpu","cuda"], help="cpu|cuda")
    ap.add_argument("--amp", action="store_true", help="CUDA autocast FP16 for speed")
    ap.add_argument("--tf32", action="store_true", help="Enable TF32 (Ampere+)")
    ap.add_argument("--csv", default="stress_results.csv", help="Where to write capacity CSV")

    # quality (real concat) flags
    ap.add_argument("--run_quality", action="store_true",
                    help="Also compute CER/WER on real concatenated data per T")
    ap.add_argument("--data_split", default="val", choices=["train","val"],
                    help="Split to draw real samples from")
    ap.add_argument("--items_per_concat", type=int, default=4,
                    help="How many items to merge for one long sample (approx)")
    ap.add_argument("--eval_batch", type=int, default=8,
                    help="Eval batch size for quality run per T")
    ap.add_argument("--quality_dir", default="stress_outputs",
                    help="Directory to write GT/Pred CSVs for quality runs")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = SimpleNamespace(**yaml.safe_load(f))

    device_str = pick_device(args.device, getattr(cfg, "device", None))
    device = torch.device(device_str)
    print(f"[INFO] Using device: {device}")

    if args.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[INFO] TF32 enabled")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model, PAD_ID, BOS_ID, EOS_ID = build_model(cfg, device)

    # load checkpoint
    if os.path.isfile(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"[OK] Loaded checkpoint: {CKPT_PATH}")
    else:
        print(f"[WARN] Fixed checkpoint not found:\n  {CKPT_PATH}\nRunning with random init weights.")

    lengths = [int(s) for s in args.lengths.split(",") if s.strip()]
    batches = [int(s) for s in args.batches.split(",") if s.strip()]
    steps_list = [int(s) for s in args.steps_list.split(",") if s.strip()]

    # -------- capacity sweep (synthetic x) --------
    header = ["length_T", "batch_B", "steps", "time_ms", "tokens_per_s", "peak_vram_MB", "status"]
    rows = []
    print("\n=== GPU decoder capacity sweep ===")
    for T in lengths:
        for B in batches:
            for steps in steps_list:
                try:
                    dt, peak = try_length(model, device, cfg.num_channel, T, BOS_ID, steps=steps, B=B, amp=args.amp)
                    tokens = B * steps
                    t_ms = dt * 1000.0
                    tps  = tokens / dt if dt > 0 else 0.0
                    peak_mb = peak / (1024**2)
                    print(f"T={T:6d} B={B:3d} steps={steps:4d}  OK  time={t_ms:7.1f} ms  tok/s={tps:9.1f}  peakVRAM={peak_mb:7.1f} MB")
                    rows.append([T, B, steps, round(t_ms,1), round(tps,1), round(peak_mb,1), "OK"])
                except RuntimeError as e:
                    msg = str(e).lower()
                    status = "OOM" if "out of memory" in msg else "FAIL"
                    print(f"T={T:6d} B={B:3d} steps={steps:4d}  {status.upper()}: {e}")
                    rows.append([T, B, steps, None, None, None, status])
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

    out_csv = os.path.abspath(args.csv)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"\n[OK] Wrote capacity sweep: {out_csv}")

    # -------- quality on real concatenated data (optional) --------
    if args.run_quality:
        os.makedirs(args.quality_dir, exist_ok=True)

        tok = None
        if getattr(cfg, "use_bpe", False) and hasattr(cfg, "tokenizer"):
            tok = BPETokenizer(cfg.tokenizer["model"])
        chars = getattr(cfg, "categories", None)

        json_name = "val.json" if args.data_split == "val" else "train.json"
        dataset_path = os.path.join(cfg.dir_dataset, json_name)
        dataset = HRDataset(dataset_path, cfg.categories, model.ratio_ds,
                            cfg.idx_fold, cfg.len_seq, cache=True)
        dataset.tokenizer = tok  # harmless if None

        print("\n=== Quality baseline on original (non-concatenated) data ===")
        # -------- baseline: original samples, no concatenation --------
        try:
            B_eval = max(1, int(args.eval_batch))
            # items_per_concat=1 → each sample is just a single original item
            x_base, len_x_base, y_ids_base = make_concat_batch(
                dataset,
                B=B_eval,
                items_per_concat=1,
                device=device,
                target_T=None,
                pad_id=(cfg.PAD_ID if hasattr(cfg, "PAD_ID") else None),
            )

            # heuristic for decode steps: ~2x longest label length
            longest_base = max(len(ids) for ids in y_ids_base) if y_ids_base else 64
            max_steps_base = max(32, 2 * longest_base)

            y_hat_base = ar_greedy_eos(
                model,
                x_base,
                len_x_base,
                BOS_ID,
                EOS_ID,
                max_steps=max_steps_base,
                amp=args.amp,
            )
            y_hat_base_cpu = y_hat_base.detach().cpu().tolist()

            base_preds, base_gts = [], []
            for b in range(len(y_ids_base)):
                pred_txt = ids_to_text(
                    y_hat_base_cpu[b][1:],  # skip BOS
                    cfg.PAD_ID,
                    cfg.BOS_ID,
                    cfg.EOS_ID,
                    chars=chars,
                    tok=tok,
                )
                gt_txt = ids_to_text(
                    y_ids_base[b],
                    cfg.PAD_ID,
                    cfg.BOS_ID,
                    cfg.EOS_ID,
                    chars=chars,
                    tok=tok,
                )
                base_preds.append(pred_txt)
                base_gts.append(gt_txt)

            base_metrics = evaluate(base_preds, base_gts)
            base_cer = base_metrics.get("character_error_rate", float("nan"))
            base_wer = base_metrics.get("word_error_rate", float("nan"))
            print(
                f"[QUALITY] T=  original  CER={base_cer:.4f}  "
                f"WER={base_wer:.4f}  (B={B_eval}, items_per_concat=0)"
            )

            base_csv_path = os.path.join(args.quality_dir, "gt_pred_Toriginal.csv")
            with open(base_csv_path, "w", newline="", encoding="utf-8") as fcsv:
                w = csv.writer(fcsv)
                w.writerow(["ground_truth", "prediction"])
                for gt, pr in zip(base_gts, base_preds):
                    w.writerow([gt, pr])
            print(f"[OK] wrote {base_csv_path}")
        except Exception as e:
            print(f"[QUALITY] baseline (original) failed: {e}")
            if device.type == "cuda":
                torch.cuda.empty_cache()

        print("\n=== Quality on real concatenated data ===")
        for T in lengths:
            try:
                B_eval = max(1, int(args.eval_batch))
                x_cat, len_x_cat, y_ids_cat = make_concat_batch(
                    dataset,
                    B=B_eval,
                    items_per_concat=args.items_per_concat,
                    device=device,
                    target_T=T,
                    pad_id=(cfg.PAD_ID if hasattr(cfg, "PAD_ID") else None),
                )
                # heuristic for decode steps: ~2x longest label length
                longest = max(len(ids) for ids in y_ids_cat) if y_ids_cat else 64
                max_steps = max(32, 2 * longest)

                y_hat = ar_greedy_eos(
                    model,
                    x_cat,
                    len_x_cat,
                    BOS_ID,
                    EOS_ID,
                    max_steps=max_steps,
                    amp=args.amp,
                )
                y_hat_cpu = y_hat.detach().cpu().tolist()

                preds, gts = [], []
                for b in range(len(y_ids_cat)):
                    pred_txt = ids_to_text(
                        y_hat_cpu[b][1:],  # skip BOS
                        cfg.PAD_ID,
                        cfg.BOS_ID,
                        cfg.EOS_ID,
                        chars=chars,
                        tok=tok,
                    )
                    gt_txt = ids_to_text(
                        y_ids_cat[b],
                        cfg.PAD_ID,
                        cfg.BOS_ID,
                        cfg.EOS_ID,
                        chars=chars,
                        tok=tok,
                    )
                    preds.append(pred_txt)
                    gts.append(gt_txt)

                metrics = evaluate(preds, gts)
                cer = metrics.get("character_error_rate", float("nan"))
                wer = metrics.get("word_error_rate", float("nan"))
                print(
                    f"[QUALITY] T={T:6d}  CER={cer:.4f}  WER={wer:.4f}  "
                    f"(B={B_eval}, items_per_concat={args.items_per_concat})"
                )

                csv_path = os.path.join(args.quality_dir, f"gt_pred_T{T}.csv")
                with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
                    w = csv.writer(fcsv)
                    w.writerow(["ground_truth", "prediction"])
                    for gt, pr in zip(gts, preds):
                        w.writerow([gt, pr])
                print(f"[OK] wrote {csv_path}")
            except Exception as e:
                print(f"[QUALITY] T={T} failed: {e}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
