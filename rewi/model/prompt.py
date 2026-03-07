# rewi/model/prompt.py
"""
Prompt Manager
==============

Manages a fixed textual instruction prompt (tokenized once) and an optional
set of learned soft-prefix vectors prepended to the LM input sequence.

The final prefix fed to the LM is:

    [soft_prefix (M tokens)]  [fixed_prompt (P tokens)]

Both are in the LM hidden dimension and are concatenated before the
modality tokens and the text tokens.

References:
    - Li & Liang, "Prefix-Tuning", arXiv:2101.00190
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PromptManager(nn.Module):
    """Fixed text prompt + optional learned soft prefix.

    Args:
        d_lm:              LM hidden dimension.
        prompt_text:       Fixed instruction string (tokenized once, frozen).
        num_soft_tokens:   Number of learned continuous prefix vectors (M).
        tokenizer:         HuggingFace tokenizer used by the LM decoder.
    """

    def __init__(
        self,
        d_lm: int,
        prompt_text: str = "Transcribe the handwritten text from IMU sensor signals:",
        num_soft_tokens: int = 20,
        tokenizer=None,
    ):
        super().__init__()
        self.d_lm = d_lm
        self.num_soft_tokens = num_soft_tokens
        self.prompt_text = prompt_text

        # Tokenize the fixed prompt and register as a non-trainable buffer
        if tokenizer is not None and prompt_text:
            ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            self.register_buffer(
                "prompt_ids", torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            )
        else:
            self.register_buffer(
                "prompt_ids", torch.zeros(1, 0, dtype=torch.long)
            )

        # Learned soft prefix  (1, M, d_lm)
        if num_soft_tokens > 0:
            self.soft_prefix = nn.Parameter(
                torch.randn(1, num_soft_tokens, d_lm) * 0.02
            )
        else:
            self.soft_prefix = None

    # ── Properties ─────────────────────────────────────────────

    @property
    def num_prompt_tokens(self) -> int:
        """Number of fixed text prompt tokens (P)."""
        return self.prompt_ids.size(1)

    @property
    def total_prefix_len(self) -> int:
        """Total prefix length: M (soft) + P (prompt)."""
        return self.num_soft_tokens + self.num_prompt_tokens

    # ── Main API ───────────────────────────────────────────────

    def get_prefix_embeds(
        self,
        embed_fn: nn.Module,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build prefix embeddings: ``[soft_prefix | text_prompt_embeds]``.

        Args:
            embed_fn:    The LM's input embedding layer (``nn.Embedding``).
            batch_size:  Current batch size.
            device:      Target device.

        Returns:
            ``(B, M + P, d_lm)`` prefix embeddings.
        """
        parts: list[torch.Tensor] = []

        # Soft prefix (learned)
        if self.soft_prefix is not None:
            parts.append(
                self.soft_prefix.to(device).expand(batch_size, -1, -1)
            )

        # Fixed text prompt (embedded through the LM's table → frozen grads flow)
        if self.num_prompt_tokens > 0:
            prompt_emb = embed_fn(self.prompt_ids.to(device))  # (1, P, d_lm)
            parts.append(prompt_emb.expand(batch_size, -1, -1))

        if parts:
            return torch.cat(parts, dim=1)  # (B, M+P, d_lm)

        return torch.zeros(batch_size, 0, self.d_lm, device=device)
