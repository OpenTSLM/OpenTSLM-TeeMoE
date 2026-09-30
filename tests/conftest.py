import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
           "out_proj", "gate_proj", "up_proj", "down_proj"]


@pytest.fixture(scope="session")
def tiny_qwen():
    """A two-layer Qwen3.5 (one recurrent, one attention layer) on CPU."""
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    for name in ("causal_conv1d_fn", "causal_conv1d_update", "FusedRMSNormGated", "chunk_gated_delta_rule",
                 "fused_recurrent_gated_delta_rule"):
        setattr(qwen, name, None)  # use the reference PyTorch implementation
    config = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, linear_num_key_heads=1, linear_num_value_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8, layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=128, attn_implementation="eager",
        rope_parameters={"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 0.5,
                         "mrope_section": [2, 1, 1]})
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(config).float().eval()
    model.requires_grad_(False)
    return model


@pytest.fixture
def experts(tiny_qwen, tmp_path):
    """Three random LoRA experts (ranks 2, 4, 3) saved as PEFT adapters."""
    from peft import LoraConfig, get_peft_model

    paths = []
    for i, rank in enumerate((2, 4, 3)):
        config = LoraConfig(r=rank, lora_alpha=2 * rank, target_modules=TARGETS)
        model = get_peft_model(copy.deepcopy(tiny_qwen), config)
        torch.manual_seed(i)
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                torch.nn.init.normal_(parameter, std=0.2)
        model.save_pretrained(tmp_path / f"expert{i}")
        paths.append(tmp_path / f"expert{i}")
    return paths
