# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This file is adapted from HuggingFace Transformers and LlamaFactory.
# It keeps only the FG_TTL training objective for easier maintenance and open-source release.
#
# Licensed under the Apache License, Version 2.0.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer with a single open-source friendly FG_TTL objective.

    Required inputs during training:
        - input_ids:      [batch, seq_len]
        - attention_mask: [batch, seq_len]
        - prompt_len:     [batch], length of the prompt part before generated tokens

    Required selector contract:
        adaptive_k_selector(per_token_entropy, attention_mask, prompt_len)
            -> mask, diagnostics

        mask:
            BoolTensor [batch, seq_len], True for selected generated-token positions.

        diagnostics["future_gains"]:
            Tensor [batch, seq_len], gain score for each token position.

    Objective:
        1. Use naive entropy without gradient to select key tokens through the selector.
        2. Compute target entropy only on selected logits to save memory.
           Supported target entropy types: naive_entropy, temperature_scaled_entropy.
        3. Weight selected-token entropy by softmax-normalized future gains.
    """

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        adaptive_k_selector=None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46") and "tokenizer" in kwargs:
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        super().__init__(**kwargs)

        # Keep a stable alias for code paths shared across Transformers versions.
        self.processing_class = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)

        if processor is not None:
            # Avoid wrong loss under gradient accumulation.
            # See: https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False
            self.add_callback(SaveProcessorCallback(processor))

        self.finetuning_args = finetuning_args
        self.adaptive_k_selector = adaptive_k_selector

        if gen_kwargs is not None:
            self._gen_kwargs = gen_kwargs

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: Optional["torch.optim.Optimizer"] = None,
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)
        return super()._get_train_sampler(*args, **kwargs)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        """Remove prompt tokens from generated tokens before metric calculation."""
        if self.args.predict_with_generate:
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model,
            inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            **gen_kwargs,
        )

        if generated_tokens is not None and self.args.predict_with_generate:
            prompt_len = inputs["input_ids"].size(-1)
            generated_tokens[:, :prompt_len] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self,
        dataset: "Dataset",
        predict_results: "PredictionOutput",
        skip_special_tokens: bool = True,
    ) -> None:
        """Save generated predictions to output_dir/generated_predictions.jsonl."""
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX,
            predict_results.label_ids,
            self.processing_class.pad_token_id,
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            non_pad = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(non_pad):
                preds[i] = np.concatenate((preds[i][non_pad[0] :], preds[i][: non_pad[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "a", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")

    @override
    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        """Compute the FG_TTL loss only."""
        if getattr(self.finetuning_args, "setting", "FG_TTL") != "FG_TTL":
            raise ValueError(
                "This open-source trainer only supports finetuning_args.setting='FG_TTL'. "
                "Please remove other settings from the config."
            )

        if self.adaptive_k_selector is None:
            raise ValueError("FG_TTL requires `adaptive_k_selector` to be provided.")

        # Do not mutate the caller-owned batch: Trainer and callbacks may reuse `inputs`.
        model_inputs = dict(inputs)
        attention_mask = model_inputs["attention_mask"]
        prompt_len = model_inputs.pop("prompt_len")

        assert getattr(self.processing_class, "padding_side", "right") == "right", (
            "FG_TTL training expects right-padded batches."
        )

        outputs = model(**model_inputs)
        logits = outputs.logits  # [batch, seq_len, vocab_size]

        # Selector entropy is used only for token selection. Keep it out of the graph.
        with torch.no_grad():
            selector_entropy = self.per_token_naive_entropy(logits.detach())  # [batch, seq_len]
            selection_mask, diagnostics = self.adaptive_k_selector(selector_entropy, attention_mask, prompt_len)

        selection_mask = selection_mask.to(device=logits.device, dtype=torch.bool)
        future_gains = diagnostics.get("future_gains") if diagnostics is not None else None
        if future_gains is None:
            raise ValueError("FG_TTL requires diagnostics['future_gains'] from adaptive_k_selector.")
        future_gains = future_gains.to(device=logits.device).detach()

        if selection_mask.shape != logits.shape[:2]:
            raise ValueError(
                f"selection_mask shape {tuple(selection_mask.shape)} must match logits[:2] {tuple(logits.shape[:2])}."
            )
        if future_gains.shape != logits.shape[:2]:
            raise ValueError(
                f"future_gains shape {tuple(future_gains.shape)} must match logits[:2] {tuple(logits.shape[:2])}."
            )

        selected_logits = logits[selection_mask]      # [num_selected, vocab_size]
        selected_gains = future_gains[selection_mask] # [num_selected]

        if selected_logits.numel() == 0:
            total_loss = logits.sum() * 0.0
        else:
            selected_entropy = self.cal_per_token_entropy(
                selected_logits,
                entropy_type=getattr(self.finetuning_args, "entropy_type", "naive_entropy"),
            )
            weights = self._gain_softmax_weights(selected_gains, dtype=selected_entropy.dtype)
            total_loss = torch.dot(weights, selected_entropy)

        self._maybe_log_fg_ttl(
            inputs=model_inputs,
            selection_mask=selection_mask,
            future_gains=future_gains,
            total_loss=total_loss,
        )

        return (total_loss, outputs) if return_outputs else total_loss

    def _gain_softmax_weights(self, selected_gains: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        gain_softmax_temp = float(getattr(self.finetuning_args, "gain_softmax_temp", 1.0))
        gain_softmax_temp = max(gain_softmax_temp, 1e-6)
        weights = F.softmax(selected_gains.float() / gain_softmax_temp, dim=0)
        return weights.to(dtype=dtype)

    def cal_per_token_entropy(
        self,
        logits: torch.Tensor,
        entropy_type: str = "naive_entropy",
    ) -> torch.Tensor:
        """Return token entropy for logits with shape [..., vocab_size]."""
        if entropy_type == "naive_entropy":
            return self.per_token_naive_entropy(logits)
        if entropy_type == "temperature_scaled_entropy":
            return self.per_token_temperature_scaled_entropy(logits)

        raise ValueError(
            "Unsupported entropy_type={!r}. Supported values: "
            "'naive_entropy', 'temperature_scaled_entropy'.".format(entropy_type)
        )

    @staticmethod
    def per_token_naive_entropy(logits: torch.Tensor) -> torch.Tensor:
        """Shannon entropy H(p) for p=softmax(logits), shape [...]."""
        log_probs = F.log_softmax(logits, dim=-1)
        return -(log_probs.exp() * log_probs).sum(dim=-1)

    def per_token_temperature_scaled_entropy(
        self,
        logits: torch.Tensor,
        alpha: Optional[float] = None,
    ) -> torch.Tensor:
        """Entropy after dynamic temperature scaling.

        T = 1 + alpha * stop_gradient(H_naive).
        Larger alpha smooths high-entropy positions more strongly.
        """
        if alpha is None:
            alpha = float(getattr(self.finetuning_args, "temperature_alpha", 0.2))

        base_entropy = self.per_token_naive_entropy(logits)
        temperature = 1.0 + alpha * base_entropy.detach()
        scaled_logits = logits / temperature.unsqueeze(-1).clamp_min(1e-6)
        return self.per_token_naive_entropy(scaled_logits)

    def _maybe_log_fg_ttl(
        self,
        inputs: dict[str, torch.Tensor],
        selection_mask: torch.Tensor,
        future_gains: torch.Tensor,
        total_loss: torch.Tensor,
    ) -> None:
        """Low-overhead optional logging. Disabled by default for speed."""
        if not self.is_world_process_zero():
            return

        log_every = int(getattr(self.finetuning_args, "dfg_log_every", 0) or 0)
        should_log_stats = log_every > 0 and self.state.global_step % log_every == 0

        if should_log_stats:
            selected_gains = future_gains[selection_mask]
            if selected_gains.numel() > 0:
                max_gain = selected_gains.max().item()
                mean_gain = selected_gains.mean().item()
            else:
                max_gain = 0.0
                mean_gain = 0.0

            log_file_path = os.path.join(self.args.output_dir, "fg_ttl_log.txt")
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(
                    f"step={self.state.global_step} "
                    f"loss={total_loss.item():.6f} "
                    f"selected={selection_mask.sum().item()} "
                    f"mean_gain={mean_gain:.6f} "
                    f"max_gain={max_gain:.6f}\n"
                )

        if bool(getattr(self.finetuning_args, "log_selected_tokens", False)):
            self.log_selected_tokens(inputs, selection_mask, prefix="fg_ttl_selected_tokens")

    def log_selected_tokens(self, inputs, mask: torch.Tensor, prefix: str = "selected_tokens") -> None:
        """Optionally write selected tokens to JSONL for debugging."""
        if not self.is_world_process_zero() or self.processing_class is None:
            return

        input_ids = inputs["input_ids"]
        log_file = os.path.join(self.args.output_dir, f"{prefix}.jsonl")

        with open(log_file, "a", encoding="utf-8") as f:
            for sample_ids, sample_mask in zip(input_ids, mask):
                targets = sample_ids[1:]
                effective_mask = sample_mask[:-1]

                min_len = min(targets.size(0), effective_mask.size(0))
                targets = targets[:min_len]
                effective_mask = effective_mask[:min_len]

                selected_ids = targets[effective_mask]
                selected_tokens = (
                    self.processing_class.batch_decode(selected_ids.unsqueeze(-1), skip_special_tokens=False)
                    if selected_ids.numel() > 0
                    else []
                )
                selected_indices = (torch.nonzero(effective_mask).flatten() + 1).tolist()

                f.write(
                    json.dumps(
                        {
                            "full_text": self.processing_class.decode(sample_ids, skip_special_tokens=False),
                            "num_selected": len(selected_tokens),
                            "selected_tokens": selected_tokens,
                            "selected_indices": selected_indices,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
