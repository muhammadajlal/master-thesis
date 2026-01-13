# rewi/model/pretrainedLM.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


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
