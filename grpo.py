import os
import re

from unsloth import is_bfloat16_supported
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

import reason_config as cfg

_NUM_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def _answer_inner_text(text: str):
    """Return the inner text of the last `<answer>...</answer>` span, if any."""
    found = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if found:
        return found[-1].strip()
    if re.search(r"<answer>", text, re.I):
        tail = text.split("<answer>")[-1]
        tail = tail.split("</answer>")[0] if "</answer>" in text else tail
        return tail.strip()
    return None


def normalize_answer(s: str) -> str:
    """Normalize numeric strings for equality (currency, commas, unicode minus)."""
    if not s:
        return ""
    s = s.strip()
    s = s.replace("$", "").replace("%", "")
    s = s.replace("\u2212", "-").replace("−", "-").replace("—", "-")
    s = re.sub(r",", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def answers_equal(gold: str, pred: str) -> bool:
    """Whether gold and predicted answers match after normalization (or as floats)."""
    g, p = normalize_answer(gold), normalize_answer(pred)
    if not g or not p:
        return False
    if g == p:
        return True
    try:
        return abs(float(g) - float(p)) < 1e-5
    except ValueError:
        return False


def extract_xml_answer(text: str) -> str:
    """Take the last number inside the answer span as the model’s final answer."""
    inner = _answer_inner_text(text)
    if inner is None:
        return ""
    raw = inner.replace(",", "")
    nums = _NUM_RE.findall(raw)
    if nums:
        return nums[-1]
    return inner.strip()


def extract_hash_answer(text):
    """Parse GSM8K gold number after `####`."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def get_gsm8k_questions(split="train"):
    """GSM8K with chat `prompt` and string gold `answer` for GRPO."""
    data = load_dataset("openai/gsm8k", "main")[split]
    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": cfg.SYS_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )
    return data


def correctness_reward_func(prompts, completions, answer, **kwargs):
    """Sparse correctness reward (+3) from parsed `<answer>` vs gold; logs one sample."""
    q = prompts[0][-1]["content"]
    gold = answer[0]
    responses = [completion[0]["content"] for completion in completions]
    extracted = [extract_xml_answer(r) for r in responses]
    print(
        f"Question:\n{q} \nAnswer:\n{gold} \nResponse:\n{responses[0]} \nExtracted:\n{extracted[0]}"
    )
    return [3.0 if answers_equal(gold, e) else 0.0 for e in extracted]


def soft_format_reward_func(completions, **kwargs):
    """Reward relaxed XML ordering: reasoning block then answer block."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [0.55 if m else 0.0 for m in matches]


def reasoning_nonempty_reward_func(completions, **kwargs):
    """Small reward if the reasoning span has a minimum length."""
    out = []
    for completion in completions:
        text = completion[0]["content"]
        m = re.search(
            r"<reasoning>(.*?)</reasoning>", text, flags=re.DOTALL | re.IGNORECASE
        )
        body = (m.group(1) if m else "").strip()
        out.append(0.2 if len(body) >= 12 else 0.0)
    return out


def run_grpo(model, tokenizer):
    """Train with GRPO, save LoRA, push merged 16-bit to Hub when `HF_TOKEN` is set."""
    dataset = get_gsm8k_questions()
    config = GRPOConfig(
        num_generations=4,
        use_vllm=True,
        learning_rate=2e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.15,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_prompt_length=256,
        max_completion_length=512,
        max_grad_norm=0.1,
        max_steps=2000,
        save_steps=500,
        report_to="wandb",
        output_dir="reason_grpo_exp",
        temperature=1.08,
        top_p=0.95,
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reasoning_nonempty_reward_func,
            soft_format_reward_func,
            correctness_reward_func,
        ],
        args=config,
        train_dataset=dataset,
    )
    trainer.train()
    model.save_lora("reason_grpo_lora")

    hf_token = os.environ.get("HF_TOKEN", "")
    model.push_to_hub_merged(
        "KickItLikeShika/Qwen2.5-1.5B-Instruct-SFT-GRPO-GSM8K",
        tokenizer,
        save_method="merged_16bit",
        token=hf_token or None,
    )
