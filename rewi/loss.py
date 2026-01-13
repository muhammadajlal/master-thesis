import torch
import torch.nn as nn


def smooth_probs(probs: torch.Tensor, alpha: float = 1e-6) -> torch.Tensor:
    '''Smooth a probability distribution with a given smoothing factor.

    Args:
        probs (torch.Tensor): Original probability distribution of shape (batch_size, len_seq, num_cls).
        alpha (float, optional): Smoothing factor. Defaults to 1e-6.

    Returns:
        torch.Tensor: Smoothed probability distribution of the same shape as input.
    '''
    num_cls = probs.shape[-1]
    distr_uni = torch.full_like(probs, 1.0 / num_cls)
    probs = (1 - alpha) * probs + alpha * distr_uni
    # ensure the smoothed probabilities sum to 1 along the last dimension
    probs /= probs.sum(dim=-1, keepdim=True)

    return probs

# original rewi loss
class CTCLoss(nn.Module):
    '''Custom CTCLoss with probability smoothing.

    Inputs:
        probs (torch.Tensor): Non-log probability predictions. (T, N, C) or (T, C) where C = number of characters in alphabet including blank, T = input length, and N = batch size.
        targets (torch.Tensor): Targets. (N, S) or (sum(target_lengths)).
        input_lengths (torch.Tensor): (N) or (). Lengths of the inputs (must each be <= T)
        target_lengths (torch.Tensor): (N) or (). Lengths of the targets.
    Outputs:
        torch.Tensor: Loss value.
    '''    
    def __init__(
        self,
        alpha_smooth: float = 1e-6,
        blank: int = 0,
        reduction: str = 'mean',
        zero_infinity: bool = False,
    ) -> None:
        '''Custom CTCLoss with probability smoothing.

        Args:
            alpha_smooth (float, optional): Smooth factor for input probability smoothing. If the factor is 0, original probability predictions are used. Defaults to 1e-6.
            blank (int, optional): Blank label. Defaults to 0.
            reduction (str, optional): Specifies the reduction to apply to the output. Options are 'none', 'mean' and 'sum'. Defaults to 'mean'.
            zero_infinity (bool, optional): Whether to zero infinite losses and the associated gradients. Defaults to False.
        '''
        super().__init__()

        self.alpha_smooth = alpha_smooth
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity

    def forward(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        '''Forward method.

        Args:
            probs (torch.Tensor): Non-log probability predictions. (T, N, C) or (T, C) where C = number of characters in alphabet including blank, T = input length, and N = batch size.
            targets (torch.Tensor): Targets. (N, S) or (sum(target_lengths)).
            input_lengths (torch.Tensor): (N) or (). Lengths of the inputs (must each be <= T)
            target_lengths (torch.Tensor): (N) or (). Lengths of the targets.

        Returns:
            torch.Tensor: Loss value.
        '''
        if self.alpha_smooth:
            probs = smooth_probs(probs, self.alpha_smooth)

        probs = probs.log()
        loss = nn.functional.ctc_loss(
            probs,
            targets,
            input_lengths,
            target_lengths,
            self.blank,
            self.reduction,
            self.zero_infinity,
        )

        return loss

"""
# original rewi loss
import math
import torch.nn.functional as F

class CTCLoss(nn.Module):

    #Expects LOGITS of shape (T, N, C) or (T, C).
    #Applies log_softmax internally. Optional prob-smoothing done in log-space.
    
    def __init__(
        self,
        alpha_smooth: float = 0.0,   # start with 0.0; you can re-enable later
        blank: int = 0,
        reduction: str = 'mean',
        zero_infinity: bool = True,  # safer to avoid NaNs when lengths mismatch
    ) -> None:
        super().__init__()
        self.alpha_smooth = float(alpha_smooth)
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity

    def forward(
        self,
        logits: torch.Tensor,        # (T, N, C) or (T, C) — LOGITS, not probs
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:

        # 1) Stable log-softmax
        log_probs = F.log_softmax(logits, dim=-1)

        # 2) Optional smoothing in log-space:
        # log( (1-a)*p + a*1/V ) = logaddexp( log(1-a)+log p , log(a)+log(1/V) )
        if self.alpha_smooth > 0.0:
            V = log_probs.size(-1)
            log_one_minus_a = math.log1p(-self.alpha_smooth)
            log_a = math.log(self.alpha_smooth)
            log_unif = -math.log(V)
            log_probs = torch.logaddexp(
                log_probs + log_one_minus_a,
                log_a + log_unif
            )

        # 3) Torch CTC expects (T, N, C)
        loss = F.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=self.blank,
            reduction=self.reduction,
            zero_infinity=self.zero_infinity,
        )
        return loss"""

