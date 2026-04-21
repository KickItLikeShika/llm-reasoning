import os

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SFT_ADAPTER_DIR = "reason_sft_lora"
SFT_NUM_SAMPLES = 100
SFT_MAX_STEPS = 100

SYS_PROMPT = """
Respond in the following format:
<reasoning>
Your short work toward the answer.
</reasoning>
<answer>
Only the final number (GSM8K answers are integers).
</answer>
"""

DEFAULT_HUB_REPO = "KickItLikeShika/Qwen2.5-1.5B-Instruct-SFT-GRPO-GSM8K"