"""Lambda scheduling for hybrid AR+CTC training."""
from typing import Any


def compute_lambda_ctc(epoch: int, schedule_cfg: dict[str, Any] | None, base_lambda: float) -> float:
    """
    Compute effective lambda_ctc for a given epoch.
    
    Args:
        epoch: Current epoch (0-indexed).
        schedule_cfg: Optional schedule config dict with keys:
            - enabled: bool (optional, default True if schedule_cfg exists)
            - type: "linear" | "cosine" (currently only linear supported)
            - warmup_epochs: int (epochs to ramp from 0 to max)
            - max: float (max lambda_ctc after warmup)
            - decay_start: int (optional, epoch to start decay)
            - min: float (optional, min lambda after decay)
        base_lambda: Fallback lambda if no schedule.
    
    Returns:
        Effective lambda_ctc for this epoch.
    """
    if schedule_cfg is None or not schedule_cfg:
        return base_lambda
    
    # Check enabled flag (default True for backward compatibility)
    if not schedule_cfg.get("enabled", True):
        return base_lambda
    
    sched_type = schedule_cfg.get("type", "linear")
    warmup_epochs = int(schedule_cfg.get("warmup_epochs", 0))
    max_lambda = float(schedule_cfg.get("max", base_lambda))
    decay_start = schedule_cfg.get("decay_start", None)
    min_lambda = float(schedule_cfg.get("min", max_lambda))
    
    # Warmup phase: linear ramp from 0 to max
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return (epoch / warmup_epochs) * max_lambda
    
    # Post-warmup: check for decay
    if decay_start is not None and epoch >= decay_start:
        # Linear decay from max to min
        decay_epochs = max(1, int(schedule_cfg.get("decay_epochs", warmup_epochs)))
        if epoch < decay_start + decay_epochs:
            progress = (epoch - decay_start) / decay_epochs
            return max_lambda + progress * (min_lambda - max_lambda)
        else:
            return min_lambda
    
    # No decay or before decay: stay at max
    return max_lambda


def balance_loss(
    loss_ar: float,
    loss_ctc: float,
    lambda_ar: float,
    lambda_ctc: float,
    balance_mode: str = "sum",
) -> float:
    """
    Combine AR and CTC losses with optional normalization.
    
    Args:
        loss_ar: AR loss value.
        loss_ctc: CTC loss value.
        lambda_ar: Weight for AR.
        lambda_ctc: Weight for CTC.
        balance_mode: "sum" (default, no normalization) | "normalize" (divide by sum of lambdas) | "convex" (treat as alpha blend).
    
    Returns:
        Combined loss.
    """
    if balance_mode == "normalize":
        total = lambda_ar + lambda_ctc
        if total > 0:
            return (lambda_ar * loss_ar + lambda_ctc * loss_ctc) / total
        else:
            return 0.0
    elif balance_mode == "convex":
        # Treat lambda_ctc as alpha: (1-alpha)*AR + alpha*CTC
        # Clamp lambda_ctc to [0,1] for safety
        alpha = max(0.0, min(1.0, lambda_ctc))
        return (1.0 - alpha) * loss_ar + alpha * loss_ctc
    else:
        # Default: sum
        return lambda_ar * loss_ar + lambda_ctc * loss_ctc
