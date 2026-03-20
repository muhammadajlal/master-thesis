# rewi/model/prompt.py
"""
Prompt Manager
==============

Manages fixed textual instruction prompts (tokenized once) and an optional
set of learned soft-prefix vectors prepended to the LM input sequence.

The final prefix fed to the LM is:

    [soft_prefix (M tokens)]  [fixed_prompt (P tokens)]

Both are in the LM hidden dimension and are concatenated before the
modality tokens and the text tokens.

**Prompt augmentation:** When a list of prompts is provided, a random prompt
is selected per call during training. During evaluation (``model.eval()``),
the first prompt (index 0) is always used for deterministic inference.

References:
    - Li & Liang, "Prefix-Tuning", arXiv:2101.00190
"""
from __future__ import annotations

import random
from typing import Union

import torch
import torch.nn as nn
from loguru import logger


class PromptManager(nn.Module):
    """Fixed text prompt(s) + optional learned soft prefix.

    Args:
        d_lm:              LM hidden dimension.
        prompt_text:       Single prompt string **or** list of prompt strings.
                           When a list is given, prompts are sampled randomly
                           during training (augmentation).
        num_soft_tokens:   Number of learned continuous prefix vectors (M).
        tokenizer:         HuggingFace tokenizer used by the LM decoder.
    """

    def __init__(
        self,
        d_lm: int,
        prompt_text: Union[str, list[str]] = "Transcribe the handwritten text from IMU sensor signals:",
        num_soft_tokens: int = 20,
        tokenizer=None,
    ):
        super().__init__()
        self.d_lm = d_lm
        self.num_soft_tokens = num_soft_tokens

        # Normalize to list
        if isinstance(prompt_text, str):
            self._prompt_texts = [prompt_text] if prompt_text else []
        else:
            self._prompt_texts = [p for p in prompt_text if p]

        # Tokenize all prompts and register as buffers
        self._num_prompts = len(self._prompt_texts)
        if tokenizer is not None and self._num_prompts > 0:
            for i, pt in enumerate(self._prompt_texts):
                ids = tokenizer.encode(pt, add_special_tokens=False)
                self.register_buffer(
                    f"prompt_ids_{i}",
                    torch.tensor(ids, dtype=torch.long).unsqueeze(0),
                )
            if self._num_prompts > 1:
                logger.info(
                    "[PromptManager] Prompt augmentation enabled: {} prompts",
                    self._num_prompts,
                )
                for i, pt in enumerate(self._prompt_texts):
                    tok_len = getattr(self, f"prompt_ids_{i}").size(1)
                    logger.info("  [{}] ({} tokens) {!r}", i, tok_len, pt)
        else:
            self.register_buffer(
                "prompt_ids_0", torch.zeros(1, 0, dtype=torch.long)
            )
            self._num_prompts = max(self._num_prompts, 1)

        # For backward compat: expose the primary prompt text
        self.prompt_text = self._prompt_texts[0] if self._prompt_texts else ""

        # Learned soft prefix  (1, M, d_lm) — shared across all prompts
        if num_soft_tokens > 0:
            self.soft_prefix = nn.Parameter(
                torch.randn(1, num_soft_tokens, d_lm) * 0.02
            )
        else:
            self.soft_prefix = None

    # ── Properties ─────────────────────────────────────────────

    def _get_prompt_ids(self, idx: int = 0) -> torch.Tensor:
        return getattr(self, f"prompt_ids_{idx}")

    @property
    def num_prompt_tokens(self) -> int:
        """Number of tokens in the primary (index-0) prompt (P)."""
        return self._get_prompt_ids(0).size(1)

    @property
    def total_prefix_len(self) -> int:
        """Total prefix length for the primary prompt: M (soft) + P (prompt)."""
        return self.num_soft_tokens + self.num_prompt_tokens

    @property
    def num_prompts(self) -> int:
        """Number of registered prompt variants."""
        return self._num_prompts

    # ── Main API ───────────────────────────────────────────────

    def get_prefix_embeds(
        self,
        embed_fn: nn.Module,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build prefix embeddings: ``[soft_prefix | text_prompt_embeds]``.

        During training with multiple prompts, a random prompt is selected
        per call (shared across the batch). During eval, prompt index 0 is
        always used.

        Args:
            embed_fn:    The LM's input embedding layer (``nn.Embedding``).
            batch_size:  Current batch size.
            device:      Target device.

        Returns:
            ``(B, M + P_i, d_lm)`` prefix embeddings.
        """
        # Select prompt index
        if self.training and self._num_prompts > 1:
            idx = random.randint(0, self._num_prompts - 1)
        else:
            idx = 0

        prompt_ids = self._get_prompt_ids(idx)
        parts: list[torch.Tensor] = []

        # Soft prefix (learned, shared across prompts)
        if self.soft_prefix is not None:
            parts.append(
                self.soft_prefix.to(device).expand(batch_size, -1, -1)
            )

        # Fixed text prompt (embedded through the LM's table)
        if prompt_ids.size(1) > 0:
            prompt_emb = embed_fn(prompt_ids.to(device))  # (1, P_i, d_lm)
            parts.append(prompt_emb.expand(batch_size, -1, -1))

        if parts:
            return torch.cat(parts, dim=1)  # (B, M+P_i, d_lm)

        return torch.zeros(batch_size, 0, self.d_lm, device=device)
