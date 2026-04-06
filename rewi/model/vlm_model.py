# rewi/model/vlm_model.py
"""
VLM Model — CNN Encoder → Q-Former → Prompt → LM
=================================================

Implements the standard VLM recipe (BLIP-2 / Flamingo / LLaVA style)
transplanted from vision→text to **IMU→text**:

1. **CNN encoder** turns raw IMU time-series into features ``(B, T, d_enc)``.
2. **Q-Former** compresses them into ``K`` modality tokens ``(B, K, d_lm)``.
3. **Fixed prompt + soft prefix** condition the decoder.
4. **LM** generates the transcript autoregressively with teacher forcing
   during training.

Supported LM backends:
- **Decoder-only** (GPT-2, LLaMA, …): conditioning is concatenated with text
  as a single ``inputs_embeds`` sequence.
- **Encoder-decoder / Seq2Seq** (T5, FlanT5, …): conditioning is passed as
  ``encoder_outputs`` and text tokens are fed to the decoder.

The model exposes the same ``forward`` / ``generate`` interface as
``MultimodalLMModel`` so the existing ``train_one_epoch_lm`` and ``test_lm``
training loops work without modification.
"""
from __future__ import annotations

import random
import string
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from transformers.modeling_outputs import BaseModelOutput

from rewi.model.projectors import build_connector
from rewi.model.prompt import PromptManager


def _build_ctc_to_lm_mapping(
    vocab_ctc: int,
    tokenizer,
    categories: list[str] | None = None,
) -> torch.Tensor:
    """Build a mapping from CTC vocabulary indices to LM token IDs.

    CTC vocab: [blank=0, char1=1, char2=2, ...]
    For each CTC class, find the corresponding LM token ID by encoding
    the single character. Blank maps to pad/eos token.

    Args:
        vocab_ctc: Number of CTC vocabulary classes.
        tokenizer: HuggingFace tokenizer.
        categories: CTC category list from config (index 0 = blank = "").
    """
    ctc_to_lm = torch.zeros(vocab_ctc, dtype=torch.long)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0

    if categories is None:
        logger.warning("[CTC→LM] No categories provided, mapping all to pad_id={}", pad_id)
        ctc_to_lm[:] = pad_id
        return ctc_to_lm

    for idx in range(min(vocab_ctc, len(categories))):
        char = categories[idx]
        if not char:  # blank
            ctc_to_lm[idx] = pad_id
        else:
            ids = tokenizer.encode(char, add_special_tokens=False)
            ctc_to_lm[idx] = ids[0] if ids else pad_id

    return ctc_to_lm


class VLMModel(nn.Module):
    """IMU Encoder → Q-Former → [soft prefix | prompt | IMU tokens] → LM.

    Supports both **decoder-only** (GPT-2, LLaMA) and **encoder-decoder**
    (T5, FlanT5) language models.

    Decoder-only mode
        Sequence layout: ``[prefix | prompt | IMU | text]`` fed as a single
        ``inputs_embeds`` tensor; loss is masked for non-text positions.

    Encoder-decoder (seq2seq) mode
        Conditioning ``[prefix | prompt | IMU]`` is passed as
        ``encoder_outputs``; text tokens are fed to the decoder via
        ``labels`` (HF shift-right creates ``decoder_input_ids``).

    Args:
        encoder:            Pre-built CNN/TCN encoder module.
        ratio_ds:           Temporal down-sampling ratio of the encoder.
        d_cnn:              Encoder output channel dimension.
        lm_name_or_path:    HuggingFace model name or local path for the
                            LM (e.g. ``gpt2``, ``t5-small``).
        connector_type:     Connector architecture: ``"qformer"``, ``"linear"``,
                            ``"mlp"``, or ``"pooling_mlp"``.
        num_queries:        Number of Q-Former output tokens (K).
        qformer_layers:     Depth of the Q-Former cross-attention stack.
        qformer_nhead:      Number of attention heads in the Q-Former.
        qformer_dropout:    Dropout inside the Q-Former Transformer.
        prompt_text:        Fixed textual instruction prompt (string or list).
        refine_prompt_text: Optional second-pass prompt template(s) used during
                            refinement decoding. Templates may include
                            ``{candidate}`` for the first-pass hypothesis.
        two_step_decode:    If ``True``, run a second refinement pass during
                            generation using ``refine_prompt_text``.
        num_soft_tokens:    Number of learned soft-prefix vectors (M).
        freeze_encoder:     If ``True``, freeze encoder weights.
        freeze_lm:          If ``True``, freeze all LM weights.
        random_init_lm:     If ``True``, randomly re-initialize LM weights
                            (control experiment to test pretrained priors).
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
        "t5": ["q", "v"],
        "default": ["q_proj", "v_proj"],
    }

    def __init__(
        self,
        encoder: nn.Module,
        ratio_ds: int,
        d_cnn: int,
        lm_name_or_path: str,
        *,
        connector_type: str = "qformer",
        num_queries: int = 32,
        qformer_layers: int = 4,
        qformer_nhead: int = 8,
        qformer_dropout: float = 0.1,
        prompt_text: str | list[str] = "Transcribe the handwritten text from IMU sensor signals:",
        refine_prompt_text: str | list[str] | None = None,
        two_step_decode: bool = False,
        num_soft_tokens: int = 20,
        freeze_encoder: bool = False,
        freeze_lm: bool = True,
        random_init_lm: bool = False,
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
        vocab_ctc: int | None = None,
        categories: list[str] | None = None,
    ):
        super().__init__()
        self.ratio_ds = int(max(1, ratio_ds))
        self.d_cnn = int(d_cnn)
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.repetition_penalty = repetition_penalty
        self.use_lora = use_lora
        self.label_smoothing = label_smoothing
        self.hybrid_mode = vocab_ctc is not None
        self.connector_type = connector_type.lower()

        # ── Encoder ─────────────────────────────────────────────
        self.encoder = encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # ── Language Model ──────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            lm_name_or_path, local_files_only=local_files_only,
        )
        # GPT-2 (and some others) lack a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Detect encoder-decoder (seq2seq) vs decoder-only
        lm_config = AutoConfig.from_pretrained(
            lm_name_or_path, local_files_only=local_files_only,
        )
        self.is_seq2seq: bool = getattr(lm_config, "is_encoder_decoder", False)

        if self.is_seq2seq:
            logger.info("[VLM] Loading seq2seq LM: {}", lm_name_or_path)
            if random_init_lm:
                logger.warning("[VLM] RANDOM INIT: re-initialising seq2seq LM from config (no pretrained weights)")
                self.lm = AutoModelForSeq2SeqLM.from_config(lm_config)
            else:
                self.lm = AutoModelForSeq2SeqLM.from_pretrained(
                    lm_name_or_path, local_files_only=local_files_only,
                )
        else:
            logger.info("[VLM] Loading causal LM: {}", lm_name_or_path)
            if random_init_lm:
                logger.warning("[VLM] RANDOM INIT: re-initialising causal LM from config (no pretrained weights)")
                self.lm = AutoModelForCausalLM.from_config(lm_config)
            else:
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
                # For seq2seq models the LM's own encoder is bypassed
                # (we feed encoder_outputs from our CNN), so restrict
                # LoRA to decoder modules only to avoid dead params.
                # PEFT uses re.fullmatch for a single regex string.
                if self.is_seq2seq:
                    alternatives = "|".join(lora_target_modules)
                    lora_target_modules = f"decoder\\..*\\.({alternatives})"

            from peft import TaskType
            task_type = (
                TaskType.SEQ_2_SEQ_LM if self.is_seq2seq
                else TaskType.CAUSAL_LM
            )
            lora_cfg = LoraConfig(
                task_type=task_type,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
            )
            self.lm = get_peft_model(self.lm, lora_cfg)
            self.lm.print_trainable_parameters()

        self.d_lm = int(self.lm.config.hidden_size)

        # ── Connector (Q-Former / Linear / MLP / Pooling) ───────
        self.connector = build_connector(
            connector_type=connector_type,
            d_enc=self.d_cnn,
            d_lm=self.d_lm,
            num_queries=num_queries,
            num_layers=qformer_layers,
            nhead=qformer_nhead,
            dropout=qformer_dropout,
            pool_tokens=num_queries,
        )
        logger.info(
            "[VLM] Connector: {} | params: {:,}",
            connector_type,
            sum(p.numel() for p in self.connector.parameters()),
        )

        # ── CTC posterior connector: set up CTC→LM token mapping ──
        if self.connector_type in ("ctc_posterior", "lego") and vocab_ctc is not None:
            from rewi.model.projectors import CTCPosteriorProjector
            if isinstance(self.connector, CTCPosteriorProjector):
                ctc_to_lm = _build_ctc_to_lm_mapping(vocab_ctc, self.tokenizer, categories)
                self.connector.set_ctc_to_lm_mapping(ctc_to_lm)
                logger.info("[VLM] CTC posterior connector: mapped {} CTC classes to LM tokens", vocab_ctc)

        # ── Prompt manager ──────────────────────────────────────
        self.prompt_manager = PromptManager(
            d_lm=self.d_lm,
            prompt_text=prompt_text,
            num_soft_tokens=num_soft_tokens,
            tokenizer=self.tokenizer,
        )

        if refine_prompt_text is None:
            self.refine_prompt_texts: list[str] = []
        elif isinstance(refine_prompt_text, str):
            self.refine_prompt_texts = [refine_prompt_text] if refine_prompt_text else []
        else:
            self.refine_prompt_texts = [p for p in refine_prompt_text if p]
        self.two_step_decode = bool(two_step_decode and self.refine_prompt_texts)
        if two_step_decode and not self.refine_prompt_texts:
            logger.warning('[VLM] two_step_decode requested but no refine_prompt_text was provided; falling back to single-pass generation')
        elif self.two_step_decode:
            logger.info('[VLM] Two-step refinement enabled: {} prompt template(s)', len(self.refine_prompt_texts))

        # ── Optional CTC auxiliary head on raw encoder features ──
        self.ctc_head: nn.Module | None = None
        if vocab_ctc is not None:
            self.ctc_head = nn.Linear(self.d_cnn, vocab_ctc)
            logger.info("[VLM] Hybrid mode: CTC head ({} → {})", self.d_cnn, vocab_ctc)

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

    def _build_runtime_prefix_embeds(
        self,
        embed_fn: nn.Module,
        prompt_text: str,
        device: torch.device,
    ) -> torch.Tensor:
        """Build ``[soft_prefix | prompt]`` for a runtime-specified prompt."""
        parts: list[torch.Tensor] = []

        if self.prompt_manager.soft_prefix is not None:
            parts.append(self.prompt_manager.soft_prefix.to(device))

        if prompt_text:
            prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            if prompt_ids:
                prompt_ids_t = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
                parts.append(embed_fn(prompt_ids_t))

        if parts:
            return torch.cat(parts, dim=1)
        return torch.zeros(1, 0, self.d_lm, device=device)

    def _decode_texts(
        self,
        output_ids: torch.Tensor,
        prompt_text: str | list[str] | None = None,
    ) -> List[str]:
        texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        if prompt_text is None:
            prompt_texts = [""] * len(texts)
        elif isinstance(prompt_text, str):
            prompt_texts = [prompt_text] * len(texts)
        else:
            prompt_texts = list(prompt_text)
            if len(prompt_texts) != len(texts):
                raise ValueError('Prompt cleanup list must match decoded batch size')

        cleaned: list[str] = []
        for t, prompt in zip(texts, prompt_texts):
            if prompt and t.startswith(prompt):
                t = t[len(prompt):].lstrip()
            cleaned.append(t)
        return cleaned

    def _generate_from_condition(
        self,
        cond_emb: torch.Tensor,
        attn_mask: torch.Tensor,
        *,
        prompt_text: str | list[str] | None = None,
    ) -> List[str]:
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
            repetition_penalty=self.repetition_penalty,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,
        )

        if self.is_seq2seq:
            encoder_outputs = BaseModelOutput(last_hidden_state=cond_emb)
            output_ids = self.lm.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=attn_mask,
                **gen_kwargs,
            )
        else:
            output_ids = self.lm.generate(
                inputs_embeds=cond_emb,
                attention_mask=attn_mask,
                **gen_kwargs,
            )

        return self._decode_texts(output_ids, prompt_text=prompt_text)

    # ── Forward (training with teacher forcing) ──────────────

    def forward(
        self,
        x: torch.Tensor,
        len_x: torch.Tensor,
        labels: torch.Tensor,
        texts: Optional[List[str]] = None,
    ):
        """Training forward pass.

        **Decoder-only** (GPT-2, LLaMA):
            Sequence: ``[prefix | prompt | IMU | text]`` fed as inputs_embeds.
            Labels:   ``[-100 × (M+P+K)] [y₁ … yL EOS]``

        **Encoder-decoder / seq2seq** (T5, FlanT5):
            Conditioning ``[prefix | prompt | IMU]`` → ``encoder_outputs``.
            Labels ``[y₁ … yL EOS]`` → passed directly to the model
            (HF handles decoder_input_ids creation via shift-right).

        Args:
            x:      ``(B, C, T_raw)`` IMU input.
            len_x:  ``(B,)`` raw input lengths.
            labels: ``(B, L)`` token IDs (text + EOS) with padding = ``-100``.
            texts:  Ground-truth strings (unused in forward, kept for API compat).

        Returns:
            HuggingFace model output with ``.loss`` and ``.logits``.
        """
        device = x.device
        B = x.size(0)

        # 1) Encode IMU
        enc_states, enc_mask = self._encode(x, len_x)  # (B, T, d_cnn), (B, T)

        # 2) Connector: compress / project encoder features
        # For CTC posterior connector, pass CTC logits and LM embeddings
        if self.connector_type in ("ctc_posterior", "lego") and self.ctc_head is not None:
            ctc_logits_for_connector = self.ctc_head(enc_states)
            embed_fn_early = self._get_embed_fn()
            lm_embed_weight = embed_fn_early.weight  # (V_lm, d_lm)
            imu_tokens = self.connector(
                enc_states, enc_mask,
                ctc_logits=ctc_logits_for_connector,
                lm_embed_weight=lm_embed_weight,
            )
        else:
            imu_tokens = self.connector(enc_states, enc_mask)  # (B, K, d_lm)
        imu_tokens = self.z_drop(imu_tokens)

        # 3) Prefix embeddings (soft prefix + fixed text prompt)
        embed_fn = self._get_embed_fn()
        prefix_emb = self.prompt_manager.get_prefix_embeds(
            embed_fn, B, device
        )  # (B, M+P, d_lm)

        if self.is_seq2seq:
            lm_out = self._forward_seq2seq(
                B, device, enc_states, imu_tokens, prefix_emb, labels,
            )
        else:
            lm_out = self._forward_causal(
                B, device, enc_states, imu_tokens, prefix_emb, labels, embed_fn,
            )

        if not self.hybrid_mode:
            if self.two_step_decode:
                # Wrap in dict so training loop can access imu_tokens
                return {"lm_out": lm_out, "imu_tokens": imu_tokens}
            return lm_out

        # Hybrid: return dict with CTC logits on raw encoder features
        T = enc_states.size(1)
        len_enc = torch.clamp(len_x // self.ratio_ds, min=1, max=T)
        result = {"lm_out": lm_out, "enc_lengths": len_enc, "imu_tokens": imu_tokens,
                  "enc_states": enc_states}
        if self.ctc_head is not None:
            result["ctc_logits"] = self.ctc_head(enc_states)  # (B, T, vocab_ctc)

        # Expose text embeddings for contrastive alignment loss
        if getattr(self, "_return_text_emb", False):
            text_ids = labels.clone()
            text_ids[text_ids == -100] = self.tokenizer.pad_token_id
            with torch.no_grad():
                text_emb = embed_fn(text_ids.to(device))  # (B, L, d_lm)
            result["text_emb"] = text_emb
            result["labels_mask"] = (labels != -100)  # (B, L) True = real token

        return result

    # ── Forward: decoder-only (causal) ──────────────────────

    def _forward_causal(
        self,
        B: int, device: torch.device,
        enc_states: torch.Tensor, imu_tokens: torch.Tensor,
        prefix_emb: torch.Tensor, labels: torch.Tensor,
        embed_fn: nn.Module,
    ):
        """Causal LM forward: [prefix | imu | text] → single inputs_embeds."""
        # 4) Text embeddings (teacher forcing)
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
                "[VLM dbg] mode=causal | enc={} → qformer={} | prefix={} "
                "(soft={}, prompt={}) | input_emb={} labels={}",
                tuple(enc_states.shape), tuple(imu_tokens.shape),
                prefix_emb.size(1), self.prompt_manager.num_soft_tokens,
                self.prompt_manager.num_prompt_tokens,
                tuple(input_emb.shape), tuple(full_labels.shape),
            )

        # 8) Forward through LM
        if self.label_smoothing > 0:
            out = self.lm(
                inputs_embeds=input_emb,
                attention_mask=attn_mask,
                labels=None,
            )
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
        return out

    # ── Forward: encoder-decoder (seq2seq) ──────────────────

    def _forward_seq2seq(
        self,
        B: int, device: torch.device,
        enc_states: torch.Tensor, imu_tokens: torch.Tensor,
        prefix_emb: torch.Tensor, labels: torch.Tensor,
    ):
        """Seq2Seq forward: conditioning → encoder_outputs, text → decoder labels.

        The conditioning ``[prefix | imu]`` is wrapped as
        ``BaseModelOutput`` and fed as ``encoder_outputs``.
        T5/FlanT5 decoder cross-attends to these embeddings.
        """
        # 4) Build conditioning = [prefix | imu]  (B, M+P+K, d_lm)
        cond_emb = torch.cat([prefix_emb, imu_tokens], dim=1)
        cond_len = cond_emb.size(1)
        cond_mask = torch.ones(B, cond_len, device=device, dtype=torch.long)
        encoder_outputs = BaseModelOutput(last_hidden_state=cond_emb)

        # Labels on device
        labels_dev = labels.to(device)

        # Debug (first batch only)
        if not self._dbg_done:
            self._dbg_done = True
            logger.info(
                "[VLM dbg] mode=seq2seq | enc={} → qformer={} | prefix={} "
                "(soft={}, prompt={}) | cond={} labels={}",
                tuple(enc_states.shape), tuple(imu_tokens.shape),
                prefix_emb.size(1), self.prompt_manager.num_soft_tokens,
                self.prompt_manager.num_prompt_tokens,
                tuple(cond_emb.shape), tuple(labels_dev.shape),
            )

        # 5) Forward through seq2seq LM
        if self.label_smoothing > 0:
            # Manually shift labels right for decoder_input_ids
            pad_id = self.tokenizer.pad_token_id
            dec_start_id = getattr(
                self.lm.config, "decoder_start_token_id", pad_id,
            )
            decoder_input_ids = labels_dev.clone()
            decoder_input_ids[:, 1:] = labels_dev[:, :-1]
            decoder_input_ids[:, 0] = dec_start_id
            decoder_input_ids[decoder_input_ids == -100] = pad_id

            out = self.lm(
                encoder_outputs=encoder_outputs,
                attention_mask=cond_mask,
                decoder_input_ids=decoder_input_ids,
                labels=None,
            )
            logits = out.logits  # (B, L, vocab)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels_dev.view(-1),
                ignore_index=-100,
                label_smoothing=self.label_smoothing,
            )
            out.loss = loss
        else:
            out = self.lm(
                encoder_outputs=encoder_outputs,
                attention_mask=cond_mask,
                labels=labels_dev,
            )
        return out

    # ── Refinement training (two-step) ────────────────────────

    @staticmethod
    def _corrupt_text(text: str, p: float = 0.3) -> str:
        """Apply random character-level corruption to simulate first-pass errors.

        With probability *p* per character, one of three mutations is applied:
        substitution (random letter), deletion, or insertion (random letter).
        """
        if not text or p <= 0:
            return text
        chars = list(text)
        result: list[str] = []
        alphabet = string.ascii_letters + "äöüÄÖÜß"
        for ch in chars:
            if random.random() < p:
                op = random.randint(0, 2)
                if op == 0:  # substitute
                    result.append(random.choice(alphabet))
                elif op == 1:  # delete
                    pass
                else:  # insert random char before
                    result.append(random.choice(alphabet))
                    result.append(ch)
            else:
                result.append(ch)
        return "".join(result) if result else text

    def forward_refine(
        self,
        imu_tokens: torch.Tensor,
        labels: torch.Tensor,
        texts: List[str],
        corrupt_prob: float = 0.3,
    ):
        """Second-pass refinement forward for two-step training.

        Reuses pre-computed ``imu_tokens`` (no re-encoding). Builds a
        refinement prompt per sample by injecting a (possibly corrupted)
        candidate into ``refine_prompt_text``, then runs a standard
        causal LM forward to produce a refinement loss.

        ~50% of the time the candidate is the clean GT (model learns to
        confirm); ~50% it is corrupted (model learns to correct).

        Args:
            imu_tokens: ``(B, K, d_lm)`` projected IMU tokens.
            labels:     ``(B, L)`` HF token IDs with -100 padding.
            texts:      Ground-truth strings for this batch.
            corrupt_prob: Per-character corruption probability.

        Returns:
            Scalar refinement loss (same scale as primary LM loss).
        """
        if not self.two_step_decode or not self.refine_prompt_texts:
            return torch.tensor(0.0, device=imu_tokens.device)

        device = imu_tokens.device
        B = imu_tokens.size(0)
        embed_fn = self._get_embed_fn()
        refine_template = self.refine_prompt_texts[0]

        # Build per-sample refinement prompts
        refine_prompts: list[str] = []
        for text in texts:
            if random.random() < 0.5:
                candidate = self._corrupt_text(text, p=corrupt_prob)
            else:
                candidate = text
            refine_prompts.append(refine_template.format(candidate=candidate))

        # Tokenize all refinement prompts and pad to same length
        all_prompt_ids = [
            self.tokenizer.encode(rp, add_special_tokens=False)
            for rp in refine_prompts
        ]
        max_prompt_len = max(len(ids) for ids in all_prompt_ids)

        # Soft prefix (shared, same as primary pass)
        parts_prefix: list[torch.Tensor] = []
        if self.prompt_manager.soft_prefix is not None:
            parts_prefix.append(
                self.prompt_manager.soft_prefix.to(device).expand(B, -1, -1)
            )
        soft_len = parts_prefix[0].size(1) if parts_prefix else 0

        # Build per-sample: [soft_prefix | refine_prompt_emb | imu_tokens | text_emb]
        # We need to pad refine prompts to the same length for batching
        padded_prompt_ids = torch.full(
            (B, max_prompt_len), self.tokenizer.pad_token_id,
            dtype=torch.long, device=device,
        )
        prompt_mask = torch.zeros(B, max_prompt_len, dtype=torch.long, device=device)
        for i, ids in enumerate(all_prompt_ids):
            padded_prompt_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            prompt_mask[i, :len(ids)] = 1

        prompt_emb = embed_fn(padded_prompt_ids)  # (B, max_prompt_len, d_lm)

        # Text embeddings (teacher forcing, same as primary)
        text_ids = labels.clone()
        text_ids[text_ids == -100] = self.tokenizer.pad_token_id
        text_emb = embed_fn(text_ids.to(device))  # (B, L, d_lm)

        # Concatenate: [soft_prefix | refine_prompt | imu_tokens | text]
        input_parts = []
        if parts_prefix:
            input_parts.append(parts_prefix[0])
        input_parts.extend([prompt_emb, imu_tokens, text_emb])
        input_emb = torch.cat(input_parts, dim=1)

        prefix_len = soft_len + max_prompt_len
        imu_len = imu_tokens.size(1)
        N = input_emb.size(1)

        # Attention mask: attend to soft prefix, valid prompt tokens, imu, valid text
        attn_mask = torch.ones(B, N, device=device, dtype=torch.long)
        # Mask padding in refine prompt region
        attn_mask[:, soft_len : soft_len + max_prompt_len] = prompt_mask
        # Mask padding in text region
        text_pad_mask = (labels == -100)
        attn_mask[:, prefix_len + imu_len :] = (~text_pad_mask).long().to(device)

        # Labels: -100 for prefix+imu, real for text
        ignore_prefix = torch.full(
            (B, prefix_len + imu_len), -100, dtype=torch.long, device=device,
        )
        full_labels = torch.cat([ignore_prefix, labels.to(device)], dim=1)

        # Forward through LM
        if self.label_smoothing > 0:
            out = self.lm(
                inputs_embeds=input_emb,
                attention_mask=attn_mask,
                labels=None,
            )
            shift_logits = out.logits[..., :-1, :].contiguous()
            shift_labels = full_labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                label_smoothing=self.label_smoothing,
            )
        else:
            out = self.lm(
                inputs_embeds=input_emb,
                attention_mask=attn_mask,
                labels=full_labels,
            )
            loss = out.loss

        return loss

    # ── Generate (inference) ──────────────────────────────────

    @torch.no_grad()
    def generate(self, x: torch.Tensor, len_x: torch.Tensor) -> List[str]:
        """Autoregressive generation from IMU input.

        Conditioning: ``[soft_prefix | prompt | imu_tokens]``.

        If ``two_step_decode`` is enabled, generation runs in two stages:
        1. Standard transcription with ``prompt_text``.
        2. Per-sample refinement using ``refine_prompt_text`` templates that
           can reference the first-pass candidate via ``{candidate}``.
        """
        device = x.device
        B = x.size(0)

        # 1) Encode + project
        enc_states, enc_mask = self._encode(x, len_x)

        # For CTC posterior connector, pass CTC logits and LM embeddings
        if self.connector_type in ("ctc_posterior", "lego") and self.ctc_head is not None:
            ctc_logits_gen = self.ctc_head(enc_states)
            embed_fn_gen = self._get_embed_fn()
            lm_embed_weight_gen = embed_fn_gen.weight
            imu_tokens = self.connector(
                enc_states, enc_mask,
                ctc_logits=ctc_logits_gen,
                lm_embed_weight=lm_embed_weight_gen,
            )
        else:
            imu_tokens = self.connector(enc_states, enc_mask)  # (B, ?, d_lm)

        # 2) First-pass decode with the configured prompt manager
        embed_fn = self._get_embed_fn()
        prefix_emb = self.prompt_manager.get_prefix_embeds(
            embed_fn, B, device
        )  # (B, M+P, d_lm)
        cond_emb = torch.cat([prefix_emb, imu_tokens], dim=1)
        attn_mask = torch.ones(B, cond_emb.size(1), device=device, dtype=torch.long)
        first_pass = self._generate_from_condition(
            cond_emb,
            attn_mask,
            prompt_text=self.prompt_manager.prompt_text,
        )

        if not self.two_step_decode:
            return first_pass

        refine_template = self.refine_prompt_texts[0]
        refined: list[str] = []
        for i, candidate in enumerate(first_pass):
            refine_prompt = refine_template.format(candidate=candidate)
            refine_prefix = self._build_runtime_prefix_embeds(
                embed_fn,
                refine_prompt,
                device,
            )
            cond_i = torch.cat([refine_prefix, imu_tokens[i : i + 1]], dim=1)
            mask_i = torch.ones(1, cond_i.size(1), device=device, dtype=torch.long)
            refined_text = self._generate_from_condition(
                cond_i,
                mask_i,
                prompt_text=refine_prompt,
            )[0].strip()
            refined.append(refined_text or candidate)

        return refined
