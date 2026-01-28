"""
Analysis module for qualitative and quantitative evaluation.

This module provides tools for:
- Attention visualization (cross-attention heatmaps)
- Grad-CAM 1D visualizations for encoder interpretability
- Sample selection based on error quantiles
- Levenshtein distance utilities
"""

from rewi.analysis.attention import CrossAttnCatcher, attn_to_matrix, save_attn_heatmap
from rewi.analysis.gradcam import GradCAM1D, seq_logprob_score, save_signal_plus_cam
from rewi.analysis.selection import (
    load_partB_selection_unified_quantile,
    load_partB_selection_by_indices,
    compute_fold_thresholds,
)
from rewi.analysis.metrics import lev_dist

__all__ = [
    # Attention
    "CrossAttnCatcher",
    "attn_to_matrix",
    "save_attn_heatmap",
    # Grad-CAM
    "GradCAM1D",
    "seq_logprob_score",
    "save_signal_plus_cam",
    # Sample selection
    "load_partB_selection_unified_quantile",
    "load_partB_selection_by_indices",
    "compute_fold_thresholds",
    # Metrics
    "lev_dist",
]
