import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    '''Custom CTCLoss with optional smoothing in log-space.

    Inputs:
        logits (torch.Tensor): Logits (not probabilities). Shape (T, N, C) or (T, C)
            where C = number of characters in alphabet including blank, T = input length,
            and N = batch size.
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
        '''Custom CTCLoss with optional smoothing in log-space.

        Args:
            alpha_smooth (float, optional): Smoothing factor in log-space. If 0, no smoothing.
                Defaults to 1e-6.
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
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        '''Forward method.

        Args:
            logits (torch.Tensor): Logits. (T, N, C) or (T, C).
            targets (torch.Tensor): Targets. (N, S) or (sum(target_lengths)).
            input_lengths (torch.Tensor): (N) or (). Lengths of the inputs (must each be <= T)
            target_lengths (torch.Tensor): (N) or (). Lengths of the targets.

        Returns:
            torch.Tensor: Loss value.
        '''
        log_probs = F.log_softmax(logits, dim=-1)

        if self.alpha_smooth and self.alpha_smooth > 0.0:
            V = log_probs.size(-1)
            log_one_minus_a = math.log1p(-self.alpha_smooth)
            log_a = math.log(self.alpha_smooth)
            log_unif = -math.log(V)
            uniform_const = torch.tensor(log_a + log_unif, device=log_probs.device, dtype=log_probs.dtype)
            log_probs = torch.logaddexp(
                log_probs + log_one_minus_a,
                uniform_const,
            )

        loss = F.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=self.blank,
            reduction=self.reduction,
            zero_infinity=self.zero_infinity,
        )

        return loss

