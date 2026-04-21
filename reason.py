import gc

from unsloth import FastLanguageModel, PatchFastRL

PatchFastRL("GRPO", FastLanguageModel)

import torch

import sft
import grpo


def release_model(model, tokenizer):
    """Detach vLLM if present, delete weights, then synchronize and flush CUDA allocators."""
    seen = set()

    def walk(m):
        if m is None or id(m) in seen:
            return
        seen.add(id(m))
        llm = getattr(m, "vllm_engine", None)
        if llm is not None:
            try:
                m.vllm_engine = None
            except Exception:
                pass
            del llm
        for name in ("model", "base_model"):
            sub = getattr(m, name, None)
            if sub is not None:
                walk(sub)

    walk(model)
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    gc.collect()


def main():
    model, tokenizer = sft.load_base_and_lora()
    sft_ds = sft.build_sft_dataset()
    sft.run_sft(model, tokenizer, sft_ds)
    release_model(model, tokenizer)

    model, tokenizer = sft.load_sft_adapter_for_grpo()
    reason_grpo.run_grpo(model, tokenizer)


if __name__ == "__main__":
    main()
