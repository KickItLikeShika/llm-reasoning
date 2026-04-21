import gc
import os

from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

import torch

import reason_config as cfg


def load_base_and_lora():
    """Load the base model in 4-bit and attach a fresh LoRA stack (same layout as grpo)."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.MODEL_NAME,
        max_seq_length=2048,
        load_in_4bit=True,
        fast_inference=False,
        max_lora_rank=64,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=32,
        use_gradient_checkpointing="unsloth",
        random_state=0,
    )
    return model, tokenizer


def split_gsm8k_answer(full: str):
    """Split GSM8K `answer` field into chain-of-thought text and final numeric line."""
    if "####" not in full:
        return full.strip(), ""
    left, right = full.rsplit("####", 1)
    return left.strip(), right.strip()


def build_sft_dataset():
    """Build a TRL dataset with conversational `prompt` / `completion` for completion-only loss."""
    raw = load_dataset("openai/gsm8k", "main")["train"]
    raw = raw.shuffle(seed=42).select(range(min(cfg.SFT_NUM_SAMPLES, len(raw))))

    def row(ex):
        trace, num = split_gsm8k_answer(ex["answer"])
        assistant = (
            f"<reasoning>\n{trace}\n</reasoning>\n<answer>\n{num.strip()}\n</answer>"
        )
        return {
            "prompt": [
                {"role": "system", "content": cfg.SYS_PROMPT},
                {"role": "user", "content": ex["question"]},
            ],
            "completion": [{"role": "assistant", "content": assistant}],
        }

    return raw.map(row, remove_columns=raw.column_names)


def run_sft(model, tokenizer, train_dataset):
    """Run SFT, then save adapter + tokenizer under `cfg.SFT_ADAPTER_DIR` and drop the trainer."""
    sft_args = SFTConfig(
        output_dir="reason_sft_runs",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_ratio=0.15,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=0.1,
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        max_steps=cfg.SFT_MAX_STEPS,
        save_steps=100,
        report_to=os.environ.get("REASON_REPORT_TO", "wandb"),
        completion_only_loss=True,
        max_length=2048,
        packing=False,
        gradient_checkpointing=True,
        max_grad_norm=0.1,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_args,
        train_dataset=train_dataset,
    )
    trainer.train()
    os.makedirs(cfg.SFT_ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(cfg.SFT_ADAPTER_DIR)
    tokenizer.save_pretrained(cfg.SFT_ADAPTER_DIR)
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def load_sft_adapter_for_grpo():
    """Reload 4-bit base plus saved SFT LoRA for the GRPO stage."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.SFT_ADAPTER_DIR,
        max_seq_length=2048,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=64,
        gpu_memory_utilization=0.7,
        enforce_eager=True,
    )
    return model, tokenizer
