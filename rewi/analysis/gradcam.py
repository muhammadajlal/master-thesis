"""
Grad-CAM 1D visualization for encoder interpretability.

Provides class activation mapping for 1D convolutional encoders to understand
which temporal regions of the input signal contribute most to predictions.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM1D:
    """
    Grad-CAM for 1D convolutional layers.
    
    Computes gradient-weighted class activation maps to visualize which
    temporal regions of the input contributed most to the prediction.
    
    Example:
        >>> cam = GradCAM1D(model, model.encoder.layers[-1].pwconv)
        >>> with torch.enable_grad():
        ...     logits = model(x, y_inp=y_inp)
        ...     score = seq_logprob_score(logits, pred_ids)
        ...     score.backward()
        >>> cam_vec = cam.cam().detach().cpu().numpy()[0]
        >>> cam.remove()
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Initialize Grad-CAM with model and target layer.
        
        Args:
            model: The full model (for reference, not directly used).
            target_layer: The Conv1d layer to compute Grad-CAM for.
        """
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.h1 = target_layer.register_forward_hook(self._forward_hook)
        self.h2 = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        """Capture activations during forward pass."""
        self.activations = out  # [B, C, T']

    def _backward_hook(self, module, gin, gout):
        """Capture gradients during backward pass."""
        self.gradients = gout[0]  # [B, C, T']

    def remove(self):
        """Remove hooks from the target layer."""
        self.h1.remove()
        self.h2.remove()

    def cam(self) -> torch.Tensor:
        """
        Compute the Grad-CAM activation map.
        
        Returns:
            Tensor of shape [B, T'] with normalized CAM values in [0, 1].
        """
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Forward and backward passes must be completed before calling cam()")
        
        acts = self.activations
        grads = self.gradients
        
        # Global average pooling of gradients
        w = grads.mean(dim=2, keepdim=True)  # [B, C, 1]
        
        # Weighted sum of activations
        cam = (w * acts).sum(dim=1)  # [B, T']
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min(dim=1, keepdim=True).values
        cam = cam / (cam.max(dim=1, keepdim=True).values + 1e-8)
        
        return cam  # [B, T']


def seq_logprob_score(logits: torch.Tensor, pred_ids: List[int]) -> torch.Tensor:
    """
    Compute sequence log-probability score for Grad-CAM backpropagation.
    
    Args:
        logits: Model output logits of shape [1, N, V] where position 0 
                predicts the first token after BOS.
        pred_ids: List of predicted token IDs (without BOS).
    
    Returns:
        Scalar tensor of summed log-probabilities (differentiable).
    """
    if len(pred_ids) == 0:
        return logits.sum() * 0.0
    
    logp = F.log_softmax(logits, dim=-1)  # [1, N, V]
    T = len(pred_ids)
    tok = torch.tensor(pred_ids, device=logits.device, dtype=torch.long).view(1, T, 1)
    gathered = torch.gather(logp[:, :T, :], dim=2, index=tok).squeeze(-1)  # [1, T]
    
    return gathered.sum()


def save_signal_plus_cam(
    x_cpu: torch.Tensor,
    cam_1d: np.ndarray,
    outpath: str,
    title: str,
    valid_len: int = None,
    figsize: tuple = (10, 3),
    dpi: int = 200,
) -> None:
    """
    Save visualization overlaying Grad-CAM on the input signal.
    
    Args:
        x_cpu: Input tensor on CPU of shape [1, C, T] or [C, T].
        cam_1d: CAM vector of shape [T'] (downsampled temporal dimension).
        outpath: Output file path for the figure.
        title: Title for the plot.
        valid_len: Optional valid length (before padding) for cropping display.
        figsize: Figure size as (width, height).
        dpi: Resolution in dots per inch.
    """
    sig = x_cpu.squeeze(0).numpy()  # [C, T]
    C, T = sig.shape

    if valid_len is None:
        valid_len = T
    valid_len = int(max(1, min(T, valid_len)))

    # Compute RMS envelope across channels
    rms = np.sqrt((sig ** 2).mean(axis=0))[:valid_len]
    
    # Upsample CAM to match signal length
    cam_up = np.interp(
        np.linspace(0, 1, valid_len),
        np.linspace(0, 1, len(cam_1d)),
        cam_1d
    )

    plt.figure(figsize=figsize)
    plt.plot(rms, linewidth=1.0, label="input (RMS across channels)")
    scale = rms.max() if rms.max() > 0 else 1.0
    plt.plot(cam_up * scale, linewidth=1.0, label="Grad-CAM (scaled)")
    plt.title(title)
    plt.xlabel("Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    plt.close()
