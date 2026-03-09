# rewi/model/vlm_model.py
"""
VLM Model — CNN Encoder → Q-Former → Prompt → Decoder-Only LM
==============================================================

Implements the standard VLM recipe (BLIP-2 / Flamingo / LLaVA style)
transplanted from vision→text to **IMU→text**:

1. **CNN encoder** turns raw IMU time-series into features ``(B, T, d_enc)``.
2. **Q-Former** compresses them into ``K`` modality tokens ``(B, K, d_lm)``.
3. **Fixed prompt + soft prefix** condition the decoder.
4. **Decoder-only LM** (GPT-2, LLaMA, …) generates the transcript
   autoregressively with teacher forcing during training.

The model exposes the same ``forward`` / ``generate`` interface as
``MultimodalLMModel`` so the existing ``train_one_epoch_lm`` and ``test_lm``
training loops work without modification.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from rewi.model.qformer import QFormerConnector
from rewi.model.prompt import PromptManager


class VLMModel(nn.Module):
    """IMU Encoder → Q-Former → [soft prefix | prompt | IMU tokens | text] → Decoder-Only LM.

    Training
        Teacher-forced AR: CE loss is computed only on transcript positions
        (prefix / IMU positions are masked with ``-100``).

    Inference
        ``generate()`` produces text conditioned on
        ``[soft prefix | prompt | IMU tokens]``.

    Args:
        encoder:            Pre-built CNN/TCN encoder module.
        ratio_ds:           Temporal down-sampling ratio of the encoder.
        d_cnn:              Encoder output channel dimension.
        lm_name_or_path:    HuggingFace model name or local path for the
                            decoder-only LM (e.g. ``gpt2``, ``TinyLlama/…``).
        num_queries:        Number of Q-Former output tokens (K).
        qformer_layers:     Depth of the Q-Former cross-attention stack.
        qformer_nhead:      Number of attention heads in the Q-Former.
        qformer_dropout:    Dropout inside the Q-Former Transformer.
        prompt_text:        Fixed textual instruction prompt.
        num_soft_tokens:    Number of learned soft-prefix vectors (M).
        freeze_encoder:     If ``True``, freeze encoder weights.
        freeze_lm:          If ``True``, freeze all LM weights.
        use_lora:           If ``True``, apply LoRA to attention projections.
        lora_r:             LoRA rank.
        lora_alpha:         LoRA alpha scaling.
        lora_dropout:       LoRA dropout.
        lora_target_modules: Which LM modules to apply LoRA to.
        max_new_tokens:     Maximum tokens generated at inference.
        num_beams:          Beam width for generation.
        repetition_penalty: Penalty for repeated tokens during generation (1.0 = off).
        local_files_only:   Enforce offline HuggingFace loading.
        z_dropout:          Dropout applied to IMU modality tokens (conditioning noise).
    """

    # Default LoRA targets per model family
    _DEFAULT_LORA_TARGETS = {
        "gpt2": ["c_attn", "c_proj"],
        "llama": ["q_proj", "v_proj"],
        "default": ["q_proj", "v_proj"],
    }

    def __init__(
        self,
        encoder: nn.Module,
        ratio_ds: int,
        d_cnn: int,
        lm_name_or_path: str,
        *,
        num_queries: int = 32,
        qformer_layers: int = 4,
        qformer_nhead: int = 8,
        qformer_dropout: float = 0.1,
        prompt_text: str = "Transcribe the handwritten text from IMU sensor signals:",
        num_soft_tokens: int = 20,
        freeze_encoder: bool = False,
        freeze_lm: bool = True,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
        max_new_tokens: int = 64,
        num_beams: int = 1,
        repetition_penalty: float = 1.0,
        local_files_only: bool = True,
        z_dropout: float = 0.1,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.ratio_ds = int(max(1, ratio_ds))
        self.d_cnn = int(d_cnn)
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.repetition_penalty = repetition_penalty
        self.use_lora = use_lora
        self.label_smoothing = label_smoothing

        # ── Encoder ─────────────────────────────────────────────
        self.encoder = encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # ── Decoder-only LM ────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            lm_name_or_path, local_files_only=local_files_only,
        )
        # GPT-2 (and some others) lack a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.lm = AutoModelForCausalLM.from_pretrained(
            lm_name_or_path, local_files_only=local_files_only,
        )

        if freeze_lm:
            for p in self.lm.parameters():
                p.requires_grad = False

        # ── LoRA (optional) ─────────────────────────────────────
        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as exc:
                raise ImportError(
                    "LoRA requires the `peft` library: pip install peft"
                ) from exc

            if lora_target_modules is None:
                # Auto-detect target modules based on model type
                model_type = getattr(self.lm.config, "model_type", "default")
                lora_target_modules = self._DEFAULT_LORA_TARGETS.get(
                    model_type, self._DEFAULT_LORA_TARGETS["default"]
                )

            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
            )
            self.lm = get_peft_model(self.lm, lora_cfg)
            self.lm.print_trainable_parameters()

        self.d_lm = int(self.lm.config.hidden_size)

        # ── Q-Former connector ──────────────────────────────────
        self.qformer = QFormerConnector(
            d_enc=self.d_cnn,
            d_lm=self.d_lm,
            num_queries=num_queries,
            num_layers=qformer_layers,
            nhead=qformer_nhead,
            dropout=qformer_dropout,
        )

        # ── Prompt manager ──────────────────────────────────────
        self.prompt_manager = PromptManager(
            d_lm=self.d_lm,
            prompt_text=prompt_text,
            num_soft_tokens=num_soft_tokens,
            tokenizer=self.tokenizer,
        )

        # ── Conditioning dropout ────────────────────────────────
        self.z_drop = nn.Dropout(z_dropout) if z_dropout > 0 else nn.Identity()

        self._dbg_done = False

    # ── Encoder helpers (shared with MultimodalLMModel) ────────

    def _call_encoder(self, x: torch.Tensor, len_x: torch.Tensor) -> torch.Tensor:
        try:
            return self.encoder(x, len_x)
        except TypeError:
            return self.encoder(x)

    def _to_BTC(self, enc_out: torch.Tensor) -> torch.Tensor:
        if enc_out.dim() != 3:
            raise ValueError(f"Expected 3D encoder output, got {tuple(enc_out.shape)}")
        if enc_out.shape[-1] == self.d_cnn:
            return enc_out
        if enc_out.shape[1] == self.d_cnn:
            return enc_out.transpose(1, 2).contiguous()
        raise ValueError(
            f"Cannot infer layout: {tuple(enc_out.shape)}, d_cnn={self.d_cnn}"
        )

    def _make_len_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        rng = torch.arange(max_len, device=lengths.device)[None, :]
        return (rng < lengths[:, None]).to(torch.long)

    def _encode(
        self, x: torch.Tensor, len_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode IMU input.

        Returns:
            enc_states: ``(B, T, d_cnn)``
            enc_mask:   ``(B, T)``  1=valid 0=pad
        """
        enc_out = self._call_encoder(x, len_x)
        enc_out = self._to_BTC(enc_out)
        B, T, C = enc_out.shape
        len_enc = torch.clamp(len_x // self.ratio_ds, min=1, max=T)
        enc_mask = self._make_len_mask(len_enc, T)
        return enc_out, enc_mask

    # ── Embedding helper ─────────────────────────────────────

    def _get_embed_fn(self) -> nn.Embedding:
        """Return the LM's input embedding layer (handles PEFT wrapper)."""
        lm = self.lm
        if hasattr(lm, "get_base_model"):
            lm = lm.get_base_model()
        if hasattr(lm, "get_input_embeddings"):
            return lm.get_input_embeddings()
        raise AttributeError("Cannot find input embeddings on LM")

    # ── Forward (training with teacher forcing) ──────────────

    def forward(
        self,
        x: torch.Tensor,
        len_x: torch.Tensor,
        labels: torch.Tensor,
        texts: Optional[List[str]] = None,
    ):
        """Training forward pass.

        Sequence layout::

            [soft_prefix (M)] [prompt (P)] [imu_tokens (K)] [y₁ … yL EOS]

        Labels::

            [-100] × (M+P+K)  [y₁ y₂ … yL EOS]

        HuggingFace internally shifts logits vs labels so that
        ``logits[last_imu]`` is trained to predict ``y₁``, etc.

        Args:
            x:      ``(B, C, T_raw)`` IMU input.
            len_x:  ``(B,)`` raw input lengths.
            labels: ``(B, L+1)`` token IDs (text + EOS) with padding = ``-100``.
            texts:  Ground-truth strings (unused in forward, kept for API compat).

        Returns:
            HuggingFace ``CausalLMOutput`` with ``.loss`` and ``.logits``.
        """
        device = x.device
        B = x.size(0)

        # 1) Encode IMU
        enc_states, enc_mask = self._encode(x, len_x)  # (B, T, d_cnn), (B, T)

        # 2) Q-Former compress
        imu_tokens = self.qformer(enc_states, enc_mask)  # (B, K, d_lm)
        imu_tokens = self.z_drop(imu_tokens)

        # 3) Prefix embeddings (soft prefix + fixed text prompt)
        embed_fn = self._get_embed_fn()
        prefix_emb = self.prompt_manager.get_prefix_embeds(
            embed_fn, B, device
        )  # (B, M+P, d_lm)

        # 4) Text embeddings (teacher forcing)
        # `labels` already has padding positions set to -100 by the collate.
        # For embedding, replace -100 with pad_token_id.
        text_ids = labels.clone()
        text_ids[text_ids == -100] = self.tokenizer.pad_token_id
        text_emb = embed_fn(text_ids.to(device))  # (B, L+1, d_lm)

        # 5) Concatenate: [prefix | imu | text]
        input_emb = torch.cat([prefix_emb, imu_tokens, text_emb], dim=1)
        prefix_len = prefix_emb.size(1)
        imu_len = imu_tokens.size(1)
        N = input_emb.size(1)

        # 6) Attention mask (1=attend, 0=ignore)
        attn_mask = torch.ones(B, N, device=device, dtype=torch.long)
        # Mask text padding positions
        text_pad_mask = (labels == -100)  # (B, L+1)
        attn_mask[:, prefix_len + imu_len :] = (~text_pad_mask).long().to(device)

        # 7) Full labels: -100 for prefix+imu positions, real IDs for text
        ignore_prefix = torch.full(
            (B, prefix_len + imu_len), -100, dtype=torch.long, device=device,
        )
        full_labels = torch.cat([ignore_prefix, labels.to(device)], dim=1)  # (B, N)

        # Debug (first batch only)
        if not self._dbg_done:
            self._dbg_done = True
            logger.info(
                "[VLM dbg] enc={} → qformer={} | prefix={} (soft={}, prompt={}) | "
                "input_emb={} labels={}",
                tuple(enc_states.shape), tuple(imu_tokens.shape),
                prefix_len, self.prompt_manager.num_soft_tokens,
                self.prompt_manager.num_prompt_tokens,
                tuple(input_emb.shape), tuple(full_labels.shape),
            )

        # 8) Forward through LM
        if self.label_smoothing > 0:
            # Get logits without HF-internal loss (pass labels=None)
            out = self.lm(
                inputs_embeds=input_emb,
                attention_mask=attn_mask,
                labels=None,
            )
            # Compute CE with label smoothing manually
            # HF convention: logits[t] predicts labels[t+1]
            shift_logits = out.logits[..., :-1, :].contiguous()
            shift_labels = full_labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                label_smoothing=self.label_smoothing,
            )
            out.loss = loss
        else:
            out = self.lm(
                inputs_embeds=input_emb,
                attention_mask=attn_mask,
                labels=full_labels,
            )
        return out  # .loss, .logits

    # ── Generate (inference) ──────────────────────────────────

    @torch.no_grad()
    def generate(self, x: torch.Tensor, len_x: torch.Tensor) -> List[str]:
        """Autoregressive generation from IMU input.

        Conditioning sequence: ``[soft_prefix | prompt | imu_tokens]``.

        Args:
            x:     ``(B, C, T_raw)`` IMU input.
            len_x: ``(B,)`` raw lengths.

        Returns:
            ``list[str]``: Decoded text predictions (one per sample).
        """
        device = x.device
        B = x.size(0)

        # 1) Encode + compress
        enc_states, enc_mask = self._encode(x, len_x)
        imu_tokens = self.qformer(enc_states, enc_mask)  # (B, K, d_lm)

        # 2) Prefix
        embed_fn = self._get_embed_fn()
        prefix_emb = self.prompt_manager.get_prefix_embeds(
            embed_fn, B, device
        )  # (B, M+P, d_lm)

        # 3) Conditioning: [prefix | imu]
        cond_emb = torch.cat([prefix_emb, imu_tokens], dim=1)  # (B, M+P+K, d_lm)
        cond_len = cond_emb.size(1)

        attn_mask = torch.ones(B, cond_len, device=device, dtype=torch.long)

        # 4) Generate
        output_ids = self.lm.generate(
            inputs_embeds=cond_emb,
            attention_mask=attn_mask,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
            repetition_penalty=self.repetition_penalty,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,
        )

        # 5) Decode — skip special tokens
        texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        # Strip leading prompt text if echoed (safety measure)
        prompt_text = self.prompt_manager.prompt_text
        cleaned: list[str] = []
        for t in texts:
            if prompt_text and t.startswith(prompt_text):
                t = t[len(prompt_text) :].lstrip()
            cleaned.append(t)

        return cleaned
