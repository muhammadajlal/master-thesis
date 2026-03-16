# rewi/model/pretrainedLM.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from loguru import logger


@dataclass
class LMConfig:
    name: str = "/home/woody/iwso/iwso214h/imu-hwr/work/REWI_work/assets/hf_models"
    train_lm: bool = False
    max_new_tokens: int = 128
    num_beams: int = 1
    length_penalty: float = 1.0
    min_new_tokens: int = 0
    local_files_only: bool = True

    # Optional decoding guards (helpful for debugging repetition)
    no_repeat_ngram_size: int = 0
    repetition_penalty: float = 1.0
    early_stopping: bool = False

    # LoRA (optional parameter-efficient fine-tuning)
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = r"decoder\..*\.(q|v)"  # regex for PEFT

    # Selectively unfreeze cross-attention key projections (modality bridge)
    unfreeze_xattn_k: bool = False


class PretrainedLMDecoder(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.name,
            local_files_only=cfg.local_files_only,
        )
        self.lm = T5ForConditionalGeneration.from_pretrained(
            cfg.name,
            local_files_only=cfg.local_files_only,
        )

        if not cfg.train_lm:
            for p in self.lm.parameters():
                p.requires_grad = False

        # LoRA adapters on decoder-only q,v attention projections
        if cfg.use_lora:
            from peft import LoraConfig, get_peft_model, TaskType

            lora_cfg = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                # Regex targets decoder attention projections (configurable scope)
                target_modules=cfg.lora_target_modules,
                bias="none",
            )
            self.lm = get_peft_model(self.lm, lora_cfg)
            self.lm.print_trainable_parameters()

        # Optionally unfreeze cross-attention key projections (EncDecAttention.k)
        # These are the most critical for bridging the modality gap: they transform
        # encoder (IMU) features into attention keys, but get no LoRA adaptation.
        if cfg.unfreeze_xattn_k:
            count = 0
            for name, param in self.lm.named_parameters():
                if "EncDecAttention.k" in name:
                    param.requires_grad = True
                    count += 1
                    logger.info("[LM] Unfreezing xattn-k: {}", name)
            logger.info("[LM] Unfroze {} cross-attention key projections", count)

    def set_trainable(self, trainable: bool) -> None:
        for p in self.lm.parameters():
            p.requires_grad = bool(trainable)

    def set_decoder_trainable(self, trainable: bool) -> None:
        """Toggle trainability for the *decoder side* of T5 (plus shared + lm_head).

        This matches the param-group selection logic used in main.py.
        """
        trainable = bool(trainable)
        for name, p in self.lm.named_parameters():
            if name.startswith("decoder.") or name.startswith("lm_head") or name.startswith("shared"):
                p.requires_grad = trainable

    @property
    def d_model(self) -> int:
        return int(self.lm.config.d_model)

    @property
    def shared_embedding(self) -> nn.Embedding:
        """Access T5's shared embedding, works with or without PEFT wrapping."""
        if hasattr(self.lm, "base_model"):
            return self.lm.base_model.model.shared
        return self.lm.shared

    def forward(self, enc_states: torch.Tensor, enc_mask: torch.Tensor, labels: torch.Tensor):
        # Ensure correct device/dtype for LM
        enc_states = enc_states.to(device=self.lm.device, dtype=self.lm.dtype)
        enc_mask = enc_mask.to(device=self.lm.device).to(torch.bool)

        encoder_outputs = BaseModelOutput(last_hidden_state=enc_states)
        out = self.lm(
            encoder_outputs=encoder_outputs,
            attention_mask=enc_mask,
            labels=labels,
        )
        return out  # out.loss, out.logits

    @torch.no_grad()
    def generate(self, enc_states: torch.Tensor, enc_mask: torch.Tensor) -> List[str]:
        # Ensure correct device/dtype for generation (no autocast by default)
        enc_states = enc_states.to(device=self.lm.device, dtype=self.lm.dtype)
        enc_mask = enc_mask.to(device=self.lm.device).to(torch.bool)

        encoder_outputs = BaseModelOutput(last_hidden_state=enc_states)

        gen_kwargs = dict(
            encoder_outputs=encoder_outputs,
            attention_mask=enc_mask,
            max_new_tokens=self.cfg.max_new_tokens,
            num_beams=self.cfg.num_beams,
            length_penalty=self.cfg.length_penalty,
            min_new_tokens=self.cfg.min_new_tokens,
        )

        if self.cfg.early_stopping:
            gen_kwargs["early_stopping"] = True
        if self.cfg.no_repeat_ngram_size and self.cfg.no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = int(self.cfg.no_repeat_ngram_size)
        if self.cfg.repetition_penalty and self.cfg.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = float(self.cfg.repetition_penalty)

        ids = self.lm.generate(**gen_kwargs)
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)
