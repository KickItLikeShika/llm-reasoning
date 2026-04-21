# LLM Reasoning
Trained Qwen2.5 1.5B model to reason to solve grade-level math with explicit structure: a short scratchpad in `<reasoning>…</reasoning>` and a single final number in `<answer>…</answer>`.
Started from pure reinforcement learning from outcome-only signals, I first followed the widely shared [willccbb GRPO demo](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb).
In practice, GRPO alone with reward tweaks didn’t work, even after several iterations on the reward, answers stayed unreliable and the XML reasoning format was broken. The model simply hadn’t seen enough correct structured completions to anchor the pattern.

I split the training in two stages:
1. Short LoRA SFT on 100 random GSM8K training examples, to teach format + roughly sensible traces, not to maximize benchmark score.
2. GRPO on top of that adapter for 2,000 steps.

SFT + GRPO behaved much better than GRPO only, the model stopped fighting the template and the credit-assignment problem got easier.

Reward Design: 
1. Small reward for non‑empty reasoning
2. Soft reward when <reasoning> and <answer> show up in order
3. Large sparse reward (+3) when the number parsed from <answer> matches the gold.

- Model Weights: https://huggingface.co/KickItLikeShika/Qwen2.5-1.5B-Instruct-SFT-GRPO-GSM8K
- W&B Report: https://wandb.ai/egyttsteam/huggingface/reports/Qwen2-5-1-5B-Reasoning-SFT-GRPO---VmlldzoxNjYxMTM3Mg

<img width="2157" height="678" alt="Screenshot 2026-04-21 at 11 18 30 AM" src="https://github.com/user-attachments/assets/ff09186e-73d6-46db-a4d6-d1bf8333e42e" />
