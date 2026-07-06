from typing import TYPE_CHECKING, List, Optional, Union

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import get_logits_processor
from ...model import load_model, load_tokenizer
from .trainer import CustomSeq2SeqTrainer
from .algrithm import (
    PerTokenDynamicFutureGainSelector
)

from ..callbacks import fix_predictor_checkpoint

import torch
import torch.nn as nn
import json
import os
import time

# === Additional imports ===
from peft import PeftModel, set_peft_model_state_dict
from safetensors.torch import load_file as safe_load_file
# ================

from ...eval.eval_utils import auto_verify
from ...eval.cal_acc import get_acc

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin
    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

from ...extras import logging
from ...extras.misc import count_parameters

logger = logging.get_logger(__name__)

class LLMTTA(nn.Module):
    def __init__(
            self, 
            data_args: "DataArguments",
            model_args: "ModelArguments", 
            training_args: "Seq2SeqTrainingArguments", 
            finetuning_args: "FinetuningArguments", 
            generating_args: "GeneratingArguments",
            tokenizer_module,
            template,
            model,
            adaptive_k_selector: Optional[Union[PerTokenDynamicFutureGainSelector]] = None,
        ):
        super().__init__()
        self.data_args = data_args
        self.training_args = training_args
        self.finetuning_args = finetuning_args
        self.model_args = model_args
        self.generating_args = generating_args
        self.template = template

        self.tokenizer_module = tokenizer_module
        self.tokenizer = self.tokenizer_module["tokenizer"]
        
        self.model = model
        self.adaptive_k_selector = adaptive_k_selector

        # Keyword arguments for `model.generate`
        gen_kwargs = self.generating_args.to_dict(obey_generation_config=True) 
        # print(f"gen_kwargs:{gen_kwargs}")
        gen_kwargs["eos_token_id"] = [self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids
        gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        gen_kwargs["logits_processor"] = get_logits_processor()
        self._gen_kwargs = gen_kwargs
        
        self.base_output_dir = self.training_args.output_dir
        self.is_first_batch = True
        
    
    def _init_trainer(self, train_dataset, **kwargs):
        data_collator = SFTDataCollatorWith4DAttentionMask(
            template=self.template,
            # pad_to_multiple_of=8 if self.training_args.do_train else None,  # for shift short attention
            pad_to_multiple_of= None,
            label_pad_token_id=IGNORE_INDEX if self.data_args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id,
            block_diag_attn=self.model_args.block_diag_attn,
            attn_implementation=getattr(self.model.config, "_attn_implementation", None),
            compute_dtype=self.model_args.compute_dtype,
            **self.tokenizer_module,
        )

        if self.finetuning_args.setting != "FG_TTL":
            raise ValueError(
                f"Only setting='FG_TTL' is supported, got: {self.finetuning_args.setting}"
            )

        TrainerCls = CustomSeq2SeqTrainer

        self.trainer = TrainerCls(
            model=self.model,
            args=self.training_args,
            finetuning_args=self.finetuning_args,
            gen_kwargs=self._gen_kwargs,
            data_collator=data_collator,
            train_dataset=train_dataset,
            adaptive_k_selector=self.adaptive_k_selector,
            **self.tokenizer_module,
            **kwargs
        )
    
    def forward(self, train_batch, predict_batch, step=1, to_save=True):
        if self.finetuning_args.setting in ["FG_TTL"]:
            self.online_forward_one_lora_all_batch(train_batch=train_batch, predict_batch=predict_batch, step=step)
            # self.online_forward_and_record_mem(train_batch=train_batch, predict_batch=predict_batch, step=step)

        else:
            raise ValueError(
                f'NO such setting: {self.finetuning_args.setting}'
            )

    def online_forward_one_lora_all_batch(self, train_batch, predict_batch, step):
        """
        Maintain a single LoRA adapter throughout the process. After each training step,
        the adapter is saved to self.base_output_dir. For the next incoming batch,
        use the trained model (as a PEFT model) to make predictions.
        After prediction, restore the base pretrained model so the newly saved adapter
        can be correctly applied for the next training iteration.
        """

        self.training_args.output_dir = self.base_output_dir + f'/predict-temperature_{self.generating_args.temperature}-max_new_tokens_{self.generating_args.max_new_tokens}'  # Directory to store prediction results

        # decoder-only models must use left-padding for batched generation.
        if self.training_args.predict_with_generate:
            self.tokenizer.padding_side = "left"  # use left-padding in generation
        
        self.training_args.generation_max_length = self.training_args.generation_max_length or self.data_args.cutoff_len
        self.training_args.generation_num_beams = self.data_args.eval_num_beams or self.training_args.generation_num_beams
        self.training_args.remove_unused_columns = False  # important for multimodal dataset
        
        self._init_trainer(train_dataset=None)
        predict_results = self.trainer.predict(predict_batch, metric_key_prefix="predict", **self._gen_kwargs)
        self.trainer.save_predictions(predict_batch, predict_results)
        
        acc = self.cal_acc_and_record(self.training_args.output_dir + '/generated_predictions.jsonl')
        if acc:
            print(f"Batch {step} accuracy: {acc}")
        
        self.unwrap_model()

        train_batch = self.get_new_train_batch(train_batch, predict_results.predictions)
        
        self.training_args.output_dir = self.base_output_dir  # Directory for saving the adapter
        self.tokenizer.padding_side = "right"

        self._init_trainer(train_dataset=train_batch)
        self.trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)
        self.trainer.save_model()    # Save the model to training_args.output_dir
        
        if self.finetuning_args.add_predictor:
            fix_predictor_checkpoint(self.model, self.training_args.output_dir, self.training_args.save_safetensors)
        self.unwrap_model()

        torch.cuda.empty_cache()  # Clear cache
        # return predict_results
    
    def online_forward_and_record_mem(self, train_batch, predict_batch, step):
        """
        Strict alignment version: preserve logical order, ACC recording, and checkpoint saving
        consistent with the reference implementation, while adding timing and memory logs.
        """
        # ==================== 1. Prediction phase setup ====================
        self.training_args.output_dir = self.base_output_dir + f'/predict-temperature_{self.generating_args.temperature}-max_new_tokens_{self.generating_args.max_new_tokens}'
        
        if self.training_args.predict_with_generate:
            self.tokenizer.padding_side = "left"
            
        self.training_args.generation_max_length = self.training_args.generation_max_length or self.data_args.cutoff_len
        self.training_args.generation_num_beams = self.data_args.eval_num_beams or self.training_args.generation_num_beams
        self.training_args.remove_unused_columns = False
        
        # --- [Start timer] full cycle ---
        start_cycle_time = time.time()

        # ==================== 2. Run prediction ====================
        self._init_trainer(train_dataset=None)
        predict_results = self.trainer.predict(predict_batch, metric_key_prefix="predict", **self._gen_kwargs)
        self.trainer.save_predictions(predict_batch, predict_results)
        
        # ==================== 3. Record accuracy ====================
        acc = self.cal_acc_and_record(self.training_args.output_dir + '/generated_predictions.jsonl')
        if acc:
            print(f"Batch {step} accuracy: {acc}")

        # Count how many new tokens were generated
        preds = predict_results.predictions
        num_gen_tokens = ((preds != self.tokenizer.pad_token_id) & (preds != -100)).sum().item()
        
        self.unwrap_model()

        # ==================== 4. Data processing ====================
        train_batch = self.get_new_train_batch(train_batch, predict_results.predictions)
        
        # [Additional stats] compute total tokens in this batch
        total_tokens_in_batch = sum(len(ids) for ids in train_batch["input_ids"])

        # ==================== 5. Training phase setup ====================
        self.training_args.output_dir = self.base_output_dir
        self.tokenizer.padding_side = "right"

        # Reset and record initial memory usage
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        memory_after_init = torch.cuda.memory_allocated()

        self._init_trainer(train_dataset=train_batch)
        
        # ==================== 6. Run training ====================
        # --- [Start timer] training only ---
        start_train_time = time.time()
        try:
            self.trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)
        finally:
            # --- [End timer] training only ---
            end_train_time = time.time()
            peak_memory_during_train = torch.cuda.max_memory_allocated()

        # --- [End timer] full cycle ---
        end_cycle_time = time.time()

        # ==================== 7. Save model ====================
        self.trainer.save_model()
        
        if self.finetuning_args.add_predictor:
            fix_predictor_checkpoint(self.model, self.training_args.output_dir, self.training_args.save_safetensors)
            
        self.unwrap_model()
        torch.cuda.empty_cache()

        # ==================== 8. Metrics computation and logging ====================
        total_duration = end_cycle_time - start_cycle_time
        train_duration = end_train_time - start_train_time
        overall_throughput = total_tokens_in_batch / total_duration if total_duration > 0 else 0

        def gb(mem):
            return mem / (1024 ** 3)

        log_file = os.path.join(self.base_output_dir, "memory_log.txt")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"Step {step} TTA full-cycle report:\n")
            f.write(f"  [Performance]\n")
            f.write(f"    Total duration: {total_duration:.2f} seconds\n")
            f.write(f"    Training duration: {train_duration:.2f} seconds\n")
            f.write(f"    Total processed tokens: {total_tokens_in_batch} (including {num_gen_tokens} generated tokens)\n")
            f.write(f"    Overall throughput: {overall_throughput:.2f} tokens/s\n")
            f.write(f"  [Resource usage]\n")
            f.write(f"    Peak memory: {gb(peak_memory_during_train):.2f} GB\n")
            f.write(f"    Extra memory usage: {gb(peak_memory_during_train - memory_after_init):.2f} GB\n")
            f.write("-" * 60 + "\n")

        print(f"[Step {step}] Overall throughput: {overall_throughput:.2f} tokens/s | Total duration: {total_duration:.2f}s | Training duration: {train_duration:.2f}s")

    def get_new_train_batch(self, train_batch, predictions):
        predictions = torch.from_numpy(predictions)
        pure_predictions = []
        prompt_lengths = [len(i) for i in train_batch['input_ids']]
        for i in range(predictions.shape[0]):
            pad_len = torch.nonzero(predictions[i] != self.tokenizer.pad_token_id, as_tuple=True)[0]
            if len(pad_len):
                start_idx = pad_len[0]
            else:
                start_idx = predictions.shape[1]

            if any(predictions[i]==-100):
                valid_end = torch.nonzero(predictions[i] != -100, as_tuple=True)[0]
            else:
                valid_end = pad_len
            
            if len(valid_end):
                end_idx = valid_end[-1]
            else:
                end_idx = -1

            if start_idx <= end_idx:
                pure_predictions.append(predictions[i][start_idx:end_idx + 1])

        prefix_length = -1
        # print(f"Truncate the first {prefix_length} tokens of the response")
        if prefix_length < 0:
            tokens_to_cat = [
                pure_predictions[idx].tolist() for idx in range(len(pure_predictions))
            ]
        else:
            tokens_to_cat = [
                pure_predictions[idx].tolist()[:prefix_length] for idx in range(len(pure_predictions))
            ]
        train_batch = train_batch.map(
            lambda x, idx: {
                **x,
                "input_ids": x["input_ids"] + tokens_to_cat[idx],
                "attention_mask": x["attention_mask"] + [1] * len(tokens_to_cat[idx]),
                "labels": x["labels"] + tokens_to_cat[idx],
                "prompt_len": prompt_lengths[idx],
            },
            with_indices=True
        )
        return train_batch

    def unwrap_model(self):
        self.model = self.trainer.accelerator.unwrap_model(self.model, keep_fp32_wrapper=False)


    def cal_acc_and_record(self, path, predict_key="predict", label_key="label"):
        # Read prediction results
        preds = []
        labels = []
        if not os.path.exists(path):
            return None
            
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                try:
                    data = json.loads(line) # use json.loads for safer parsing
                    preds.append(data[predict_key])
                    labels.append(data[label_key])
                except Exception:
                    continue

        dataset_type = None
        if "gsm8k_test_formatted" in self.data_args.dataset:
            dataset_type = "gsm8k1"
        elif "gsm8k_test_add_suffix" in self.data_args.dataset:
            dataset_type = "gsm8k2"
        elif "math_500" in self.data_args.dataset:
            dataset_type = "math_500"
        elif "mawps_formatted" in self.data_args.dataset:
            dataset_type = "mawps"
        elif "college_math_formatted" in self.data_args.dataset:
            dataset_type = "college_math"
        elif any("aime" in ds for ds in self.data_args.dataset):
            dataset_type = "aime"
        elif any("minerva" in ds for ds in self.data_args.dataset):
            dataset_type = "minerva"
        elif any("math_vision" in ds for ds in self.data_args.dataset):
            dataset_type = "math_vision"

        if dataset_type:
            all_results = auto_verify(preds, labels, dataset_type)
            if len(all_results) > 0:
                acc = sum(all_results) / len(all_results)
                message = f"总共 {len(all_results)} 个样本，准确率为：{acc}"
                print(message)
                with open(os.path.dirname(path) + '/acc_recorder', 'a', encoding='utf-8') as f:
                    f.write(message + '\n')
                return acc
        return None


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    print(f"{data_args}\n{model_args}\n{training_args}")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    train_dataset = dataset_module['train_dataset']
    eval_dataset = dataset_module['eval_dataset']
    print(f"Train Dataset Size: {len(train_dataset)}")
    num_samples = len(train_dataset)

    # Initialize selector
    if finetuning_args.setting != "FG_TTL":
        raise ValueError(
            f"Only setting='FG_TTL' is supported, got: {finetuning_args.setting}"
        )

    selector = PerTokenDynamicFutureGainSelector()

    llm_tta = LLMTTA(
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        generating_args=generating_args,
        tokenizer_module=tokenizer_module,
        template=template,
        model=model,
        adaptive_k_selector=selector
    )

    batch_size = getattr(finetuning_args, "tta_batch_size", 8)

    num_of_batch = len(train_dataset) // batch_size
    if len(train_dataset) % batch_size != 0:
        num_of_batch += 1
    
    # ==================== Resume-from-checkpoint logic starts ====================
    start_batch_idx = 0
    
    # Strategy: read generated_predictions.jsonl
    # Build the prediction file path
    pred_dir_name = f"predict-temperature_{generating_args.temperature}-max_new_tokens_{generating_args.max_new_tokens}"
    pred_file_path = os.path.join(training_args.output_dir, pred_dir_name, "generated_predictions.jsonl")
    
    if os.path.exists(pred_file_path):
        try:
            with open(pred_file_path, "r", encoding="utf-8") as f:
                processed_samples = sum(1 for _ in f)
            calculated_idx = processed_samples // batch_size
            start_batch_idx = calculated_idx
            
            if start_batch_idx > 0:
                logger.info(f"Found prediction file with {processed_samples} samples. Resuming from batch index: {start_batch_idx}")
        except Exception as e:
            logger.warning(f"Failed to count lines in prediction file: {e}.")

    # Load the latest adapter weights (for resume-from-checkpoint)
    if start_batch_idx > 0:
        adapter_path = training_args.output_dir
        adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
        # Check whether weight files exist
        has_bin = os.path.exists(os.path.join(adapter_path, "adapter_model.bin"))
        has_safe = os.path.exists(os.path.join(adapter_path, "adapter_model.safetensors"))
        
        if os.path.exists(adapter_config_path) and (has_bin or has_safe):
            logger.info(f"Loading evolved adapter from {adapter_path}...")
            try:
                # Load weights using set_peft_model_state_dict
                if has_safe:
                    adapters_weights = safe_load_file(os.path.join(adapter_path, "adapter_model.safetensors"))
                else:
                    adapters_weights = torch.load(os.path.join(adapter_path, "adapter_model.bin"), map_location="cpu")
                
                set_peft_model_state_dict(model, adapters_weights, adapter_name="default")
                logger.info("Successfully loaded evolved adapter weights.")
            except Exception as e:
                logger.error(f"Failed to load adapter weights: {e}. Training might be compromised.")
                start_batch_idx = 0 # Reset if weight loading fails
        else:
            logger.warning(f"Found progress ({start_batch_idx}) but NO ADAPTER WEIGHTS found in {adapter_path}. "
                           "Cannot resume training state (model will be reset to base). "
                           "Starting from batch 0 to ensure consistency.")
            start_batch_idx = 0
    # ==================== Resume-from-checkpoint logic ends ====================

    import time
    start = time.time()

    for k in range(start_batch_idx, num_of_batch):
        logger.info(f"Processing batch {k+1} / {num_of_batch}")
        
        if (k+1)*batch_size > len(train_dataset):
            end_index = len(train_dataset)
        else:
            end_index = (k+1)*batch_size
            
        sub_trainset = train_dataset.select(range(k*batch_size, end_index))
        sub_evalset = eval_dataset.select(range(k*batch_size, end_index))
        
        llm_tta.forward(sub_trainset, sub_evalset, step=k+1)

    end = time.time()
    logger.info(f"Dataset {data_args.dataset} processing complete, total samples processed: {num_samples}.")
    logger.info(f"Total time: {end - start} seconds. Average time per sample: {(end - start) / num_samples} seconds.")