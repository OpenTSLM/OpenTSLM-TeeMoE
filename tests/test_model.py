import copy

import torch

from teemoe.model.editor import AggregationEditor, editor_loss
from teemoe.model.mixture import attach_experts, bake_mixture, convert_for_vllm, expert_weights, read_lora


def test_mixture_matches_scaled_peft_adapters(tiny_qwen, experts):
    from peft import PeftModel
    from peft.tuners.lora.layer import LoraLayer

    ids = torch.randint(0, 64, (2, 9))
    weights = torch.tensor([[0.2, 0.5, 0.3], [0.7, 0.1, 0.2]])
    reference = PeftModel.from_pretrained(copy.deepcopy(tiny_qwen), experts[0], adapter_name="0")
    for i in (1, 2):
        reference.load_adapter(experts[i], adapter_name=str(i))
    reference.base_model.set_adapter(["0", "1", "2"])
    expected = []
    for row in range(2):
        for layer in reference.modules():
            if isinstance(layer, LoraLayer):
                for key in "012":
                    layer.scaling[key] = weights[row, int(key)].item() * layer.lora_alpha[key] / layer.r[key]
        expected.append(reference(input_ids=ids[row:row + 1]).logits)
    mixed = copy.deepcopy(tiny_qwen)
    attach_experts(mixed, experts)
    with expert_weights(mixed, weights):
        assert torch.allclose(mixed(input_ids=ids).logits, torch.cat(expected), atol=1e-5)
    with expert_weights(mixed, None):
        assert torch.allclose(mixed(input_ids=ids).logits, tiny_qwen(input_ids=ids).logits, atol=1e-6)


def test_baked_mixture_is_one_equivalent_lora(tiny_qwen, experts, tmp_path):
    from peft import PeftModel

    ids = torch.randint(0, 64, (1, 9))
    weights = [0.2, 0.5, 0.3]
    baked = bake_mixture(experts, weights, tmp_path / "baked", dtype=torch.float32)
    single = PeftModel.from_pretrained(copy.deepcopy(tiny_qwen), baked)
    mixed = copy.deepcopy(tiny_qwen)
    attach_experts(mixed, experts)
    with expert_weights(mixed, torch.tensor([weights])):
        assert torch.allclose(single(input_ids=ids).logits, mixed(input_ids=ids).logits, atol=1e-5)
    config = tiny_qwen.config.to_dict()
    converted, _, _ = read_lora(convert_for_vllm(baked, tmp_path / "vllm", config))
    names = [name for name in converted if "linear_attn.in_proj_" in name]
    assert any(name.endswith("in_proj_q") for name in names) and not any(n.endswith("in_proj_qkv") for n in names)


def test_editor_starts_at_the_reference(tiny_qwen):
    editor = AggregationEditor(32)
    candidates = torch.sort(torch.randn(2, 13, 11, 9) + 5, dim=-1).values
    reference = torch.sort(torch.randn(2, 11, 9) + 5, dim=-1).values
    refined, correction = editor(tiny_qwen.model, candidates, reference)
    assert torch.equal(correction, torch.zeros_like(correction))
    assert torch.allclose(refined, reference) and torch.allclose(editor.predict(tiny_qwen.model, candidates, reference),
                                                                 reference)
    loss = editor_loss(correction, target=reference[..., 4] + 1, reference=reference, candidates=candidates,
                       knot_weight=torch.ones(2, 11))
    assert loss.rows == 2 and torch.isfinite(loss.total)
