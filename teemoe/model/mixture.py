"""Compose frozen LoRA experts with request-level mixture weights.

For an adapted matrix W, a request with expert weights pi computes
    W' x = W x + sum_e pi_e * (alpha_e / r_e) * B_e A_e x,
with one weight vector per request, shared by all layers and held fixed for
the whole response. Weights are tensors, so the controller can be trained
through them.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

EXPERTS = ("aggregation", "native", "analysis")
PEFT_PREFIX = "base_model.model."


def read_lora(path: str | Path) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], float, dict]:
    """Read a saved PEFT LoRA as {module path: (A, B)}, its scale alpha/r and its config."""
    path = Path(path)
    config = json.loads((path / "adapter_config.json").read_text())
    if config.get("use_dora") or config.get("use_rslora") or config.get("rank_pattern") \
            or config.get("alpha_pattern"):
        raise ValueError(f"{path}: only ordinary constant-rank LoRA adapters are supported")
    state = load_file(str(path / "adapter_model.safetensors"))
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state.items():
        module, _, kind = key.removeprefix(PEFT_PREFIX).rpartition(".lora_")
        pairs.setdefault(module, {})[kind.split(".")[0]] = value
    if any(set(value) != {"A", "B"} for value in pairs.values()):
        raise ValueError(f"{path}: every adapted module needs lora_A and lora_B")
    return {m: (v["A"], v["B"]) for m, v in pairs.items()}, config["lora_alpha"] / config["r"], config


class MixtureState:
    """Mixture weights shared by every mixed layer; ``None`` disables all experts."""

    def __init__(self) -> None:
        self.weights: torch.Tensor | None = None


class MixedLoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, state: MixtureState) -> None:
        super().__init__()
        self.base, self.state, self.scales = base, state, []

    def add_expert(self, a: torch.Tensor, b: torch.Tensor, scale: float) -> None:
        index = len(self.scales)
        device = self.base.weight.device
        self.register_buffer(f"lora_A{index}", a.float().to(device), persistent=False)
        self.register_buffer(f"lora_B{index}", b.float().to(device), persistent=False)
        self.scales.append(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        weights = self.state.weights
        if weights is None:
            return out
        if weights.shape[0] != x.shape[0]:  # e.g. several samples per request
            weights = weights.repeat_interleave(x.shape[0] // weights.shape[0], dim=0)
        total = out.float()
        hidden = x.float()
        shape = (-1,) + (1,) * (x.dim() - 1)
        for index, scale in enumerate(self.scales):
            a, b = getattr(self, f"lora_A{index}"), getattr(self, f"lora_B{index}")
            if a.numel():
                total = total + F.linear(F.linear(hidden, a), b) * (weights[:, index] * scale).view(shape)
        return total.to(out.dtype)


def attach_experts(model: torch.nn.Module, adapters: list[str | Path]) -> MixtureState:
    """Wrap every adapted linear layer of ``model`` with the given frozen experts, in order."""
    state = MixtureState()
    loras = [read_lora(path) for path in adapters]
    modules = sorted(set().union(*(set(pairs) for pairs, _, _ in loras)))
    for name in modules:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        base = getattr(parent, child)
        if not isinstance(base, torch.nn.Linear):
            raise TypeError(f"{name} is not a linear layer")
        mixed = MixedLoRALinear(base, state)
        for pairs, scale, _ in loras:
            a, b = pairs.get(name, (torch.empty(0), torch.empty(0)))
            mixed.add_expert(a, b, scale)
        setattr(parent, child, mixed)
    model._teemoe_mixture = state
    return state


@contextmanager
def expert_weights(model: torch.nn.Module, weights: torch.Tensor | None):
    """Run ``model`` with per-request expert weights [requests, experts] (``None``: base model)."""
    state: MixtureState = model._teemoe_mixture
    previous, state.weights = state.weights, weights
    try:
        yield
    finally:
        state.weights = previous


def mixture_tensors(loras: list, weights, dtype: torch.dtype = torch.bfloat16) -> tuple[dict, dict]:
    """One PEFT LoRA (tensors, config) equal to the weighted mixture of loaded experts (``read_lora``
    results): ranks are concatenated, and B carries each expert's weight and scale."""
    weights = torch.as_tensor(weights, dtype=torch.float32)
    if weights.shape != (len(loras),):
        raise ValueError("one weight per expert is required")
    tensors = {}
    for name in sorted(set().union(*(set(pairs) for pairs, _, _ in loras))):
        a_parts, b_parts = [], []
        for (pairs, scale, _), weight in zip(loras, weights, strict=True):
            if name in pairs:
                a, b = pairs[name]
                a_parts.append(a.float())
                b_parts.append(b.float() * (weight * scale))
        tensors[f"{PEFT_PREFIX}{name}.lora_A.weight"] = torch.cat(a_parts, 0).to(dtype).contiguous()
        tensors[f"{PEFT_PREFIX}{name}.lora_B.weight"] = torch.cat(b_parts, 1).to(dtype).contiguous()
    rank = sum(config["r"] for _, _, config in loras)
    return tensors, dict(loras[0][2], r=rank, lora_alpha=rank, lora_dropout=0.0, inference_mode=True)


def save_lora(tensors: dict, config: dict, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(destination / "adapter_model.safetensors"))
    (destination / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    return destination


def bake_mixture(adapters: list[str | Path], weights, destination: str | Path,
                 dtype: torch.dtype = torch.bfloat16) -> Path:
    """Write one LoRA equal to the weighted expert mixture (ranks concatenated), e.g. for vLLM."""
    return save_lora(*mixture_tensors([read_lora(path) for path in adapters], weights, dtype), destination)


def vllm_names(tensors: dict, text_config: dict) -> dict:
    """Rename Qwen3.6 LoRA tensors for vLLM, which splits the fused recurrent QKV projection."""
    q = int(text_config["linear_num_key_heads"]) * int(text_config["linear_key_head_dim"])
    v = int(text_config["linear_num_value_heads"]) * int(text_config["linear_value_head_dim"])
    renamed = {}
    for key, value in tensors.items():
        key = key.replace(f"{PEFT_PREFIX}model.layers.", f"{PEFT_PREFIX}language_model.model.layers.")
        if ".in_proj_qkv." not in key:
            renamed[key] = value
            continue
        pieces = ([value.clone() for _ in range(3)] if ".lora_A." in key
                  else value.split([q, q, v], dim=0))
        for part, piece in zip(("in_proj_q", "in_proj_k", "in_proj_v"), pieces, strict=True):
            renamed[key.replace("in_proj_qkv", part)] = piece.contiguous()
    return renamed


def convert_for_vllm(adapter: str | Path, destination: str | Path, text_config: dict) -> Path:
    """Rename a saved Qwen3.6 LoRA for vLLM (see ``vllm_names``)."""
    adapter = Path(adapter)
    tensors = vllm_names(load_file(str(adapter / "adapter_model.safetensors")), text_config)
    return save_lora(tensors, json.loads((adapter / "adapter_config.json").read_text()), destination)
