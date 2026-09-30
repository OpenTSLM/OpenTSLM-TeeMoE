"""Loading the shared Qwen3.6-27B backbone and its tokenizer."""

from __future__ import annotations

import torch

BASE_MODEL = "Qwen/Qwen3.6-27B"
BASE_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
# Every expert adapts the same attention, recurrent-mixing and feed-forward projections.
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z",
                  "in_proj_b", "in_proj_a", "out_proj", "gate_proj", "up_proj", "down_proj")


def load_tokenizer(model_id: str = BASE_MODEL, revision: str | None = BASE_REVISION):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_base_model(model_id: str = BASE_MODEL, revision: str | None = BASE_REVISION, *,
                    device: str | torch.device = "cuda", gradient_checkpointing: bool = False):
    """Load the frozen BF16 backbone on one device."""
    from transformers import AutoModelForCausalLM

    device = torch.device(device)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        device_map={"": device.index or 0} if device.type == "cuda" else None)
    if device.type == "cuda":
        require_fast_kernels(model)
    model.requires_grad_(False)
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return model


def require_fast_kernels(model) -> None:
    """Qwen3.6's recurrent layers are impractically slow without FLA and causal-conv1d."""
    if not str(model.config.model_type).startswith("qwen3_5"):
        return
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    if not qwen.is_fast_path_available:
        raise RuntimeError("Install the accelerated kernels with `bash scripts/setup.sh inference` "
                           "to run Qwen3.6 on GPU.")


def decoder(model):
    """The transformer stack (without the language-model head) of a causal LM or PEFT wrapper."""
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    return model.get_decoder() if hasattr(model, "get_decoder") else model.model
