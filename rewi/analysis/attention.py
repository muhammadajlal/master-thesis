"""
Attention visualization utilities for decoder cross-attention analysis.

Provides tools to capture and visualize attention weights from transformer decoders.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class CrossAttnCatcher:
    """
    Captures attention weights from decoder cross-attention modules.
    
    Works with standard PyTorch MultiheadAttention layers named 'multihead_attn'
    (i.e., decoder cross-attention, not self-attention).
    
    Example:
        >>> catcher = CrossAttnCatcher()
        >>> catcher.patch_decoder_cross_attn(model.decoder)
        >>> with torch.no_grad():
        ...     _ = model(x, y_inp=y_inp)
        >>> M = attn_to_matrix(catcher.weights)
        >>> catcher.unpatch()
    """
    
    def __init__(self):
        self.weights: List[torch.Tensor] = []
        self.handles = []
        self.patched = []

    def clear(self):
        """Clear captured attention weights."""
        self.weights.clear()

    def hook(self, module, inp, out):
        """Forward hook to capture attention weights."""
        # MultiheadAttention returns (attn_output, attn_weights) if need_weights=True
        if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
            self.weights.append(out[1].detach().cpu())

    def patch_decoder_cross_attn(self, decoder: nn.Module):
        """
        Patch ONLY cross-attention modules to force need_weights=True.
        
        This modifies the forward method of MultiheadAttention layers
        to capture attention weights during inference.
        
        Args:
            decoder: The decoder module containing cross-attention layers.
        """
        for name, m in decoder.named_modules():
            # We only want cross-attention, not self-attention
            if isinstance(m, nn.MultiheadAttention) and "multihead_attn" in name and "self_attn" not in name:
                orig_forward = m.forward

                def make_wrapped_forward(original):
                    def wrapped_forward(*args, **kwargs):
                        kwargs["need_weights"] = True
                        # Keep per-head weights if available
                        if "average_attn_weights" in kwargs:
                            kwargs["average_attn_weights"] = False
                        return original(*args, **kwargs)
                    return wrapped_forward

                m.forward = make_wrapped_forward(orig_forward)
                self.patched.append((m, orig_forward))
                self.handles.append(m.register_forward_hook(self.hook))

    def unpatch(self):
        """Remove hooks and restore original forward methods."""
        for h in self.handles:
            h.remove()
        self.handles.clear()
        for m, orig in self.patched:
            m.forward = orig
        self.patched.clear()


def attn_to_matrix(attn_list: List[torch.Tensor], expected_tgt_len: int = None) -> Optional[np.ndarray]:
    """
    Combine list of attention tensors into a single [tgt_len, src_len] matrix.
    
    Handles common shapes:
      - [B, heads, tgt, src]
      - [B, tgt, src]
      - [tgt, src]
    
    Args:
        attn_list: List of attention tensors from different layers.
        expected_tgt_len: Optional expected target length to validate/transpose result.
    
    Returns:
        Normalized attention matrix of shape [tgt_len, src_len], or None if empty.
    """
    if not attn_list:
        return None

    mats = []
    for a in attn_list:
        t = a
        while t.dim() > 2:
            t = t.mean(dim=0)  # Average batch/heads progressively
        mats.append(t)

    M = torch.stack(mats, dim=0).mean(dim=0)  # [tgt, src] ideally
    
    # Normalize to [0, 1]
    M = M - M.min()
    if M.max() > 0:
        M = M / M.max()

    M = M.numpy()

    # If expected target length is given, ensure first dimension matches it
    if expected_tgt_len is not None:
        # Common case: M is [src, tgt] -> transpose
        if M.shape[0] != expected_tgt_len and M.shape[1] == expected_tgt_len:
            M = M.T
    
    return M


def save_attn_heatmap(
    M: np.ndarray,
    outpath: str,
    title: str,
    figsize: tuple = (8, 4),
    dpi: int = 200,
    cmap: str = "viridis",
) -> None:
    """
    Save attention matrix as a heatmap visualization.
    
    Args:
        M: Attention matrix of shape [tgt_len, src_len].
        outpath: Output file path for the figure.
        title: Title for the plot.
        figsize: Figure size as (width, height).
        dpi: Resolution in dots per inch.
        cmap: Matplotlib colormap name.
    """
    plt.figure(figsize=figsize)
    plt.imshow(M, aspect="auto", origin="lower", cmap=cmap)
    plt.colorbar(label="attention")
    plt.xlabel("Encoder time position (downsampled)")
    plt.ylabel("Token position")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    plt.close()
