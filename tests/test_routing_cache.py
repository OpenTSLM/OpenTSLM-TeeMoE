from collections import OrderedDict
from types import SimpleNamespace

import torch

from teemoe.model import controller
from teemoe.teemoe import TeeMoE


def fake_encoder(monkeypatch):
    calls = []

    def last_state(model, ids, device):
        calls.append(tuple(ids))
        return torch.tensor([sum(ids), len(ids)], dtype=torch.float32, device=device)

    monkeypatch.setattr(controller, "last_state", last_state)
    monkeypatch.setattr(controller, "routing_view", lambda prompt: prompt.split("|")[0])
    tokenizer = SimpleNamespace(encode=lambda value, **kw: [ord(c) for c in value])
    return torch.nn.Linear(1, 1), tokenizer, calls


def test_identical_views_are_cached_across_batches(monkeypatch):
    model, tokenizer, calls = fake_encoder(monkeypatch)
    cache = OrderedDict()
    prompts = [f"forecast|signal-{i}" for i in range(1000)]
    text, full = controller.request_states(model, tokenizer, prompts, full=False, cache=cache)
    assert len(calls) == 1 and torch.equal(text, text[:1].expand_as(text))
    assert not full.any()
    again, _ = controller.request_states(model, tokenizer, prompts[:2], full=False, cache=cache)
    assert len(calls) == 1 and torch.equal(again, text[:2])
    assert all(value.device.type == "cpu" for value in cache.values())


def test_full_requests_keep_their_distinct_signal_states(monkeypatch):
    model, tokenizer, calls = fake_encoder(monkeypatch)
    prompts = ["forecast|1", "forecast|9", "forecast|1"]
    cache = OrderedDict()
    text, full = controller.request_states(model, tokenizer, prompts, cache=cache)
    assert len(calls) == 3  # one short view and two full views
    assert torch.equal(text[0], text[1])
    assert not torch.equal(full[0], full[1]) and torch.equal(full[0], full[2])
    again = controller.request_states(model, tokenizer, list(reversed(prompts)), cache=cache)
    assert len(calls) == 3
    assert torch.equal(again[0], text.flip(0)) and torch.equal(again[1], full.flip(0))


def test_state_cache_is_bounded_and_least_recently_used(monkeypatch):
    model, tokenizer, calls = fake_encoder(monkeypatch)
    monkeypatch.setattr(controller, "STATE_CACHE_SIZE", 2)
    cache = OrderedDict()
    controller.request_states(model, tokenizer, ["a", "b", "a", "c"], full=False, cache=cache)
    assert list(cache) == [(ord("a"),), (ord("c"),)] and len(calls) == 3
    controller.request_states(model, tokenizer, ["a", "b"], full=False, cache=cache)
    assert len(cache) == 2 and len(calls) == 4


def test_teemoe_cache_reuses_features_not_controller_weights(monkeypatch):
    model, tokenizer, calls = fake_encoder(monkeypatch)
    instance = TeeMoE.__new__(TeeMoE)
    instance.model, instance.tokenizer = model, tokenizer
    instance.controller = controller.Controller(2)
    instance._routing_cache = OrderedDict()
    prompts = ["forecast|1", "forecast|9"]
    instance.route(prompts)
    instance.route(prompts)
    assert len(calls) == 1
    with torch.no_grad():
        instance.controller.signal_fraction.fill_(0.25)
        instance.controller.head.weight[0].fill_(0.001)
    actual = instance.route(prompts)
    text, full = controller.request_states(model, tokenizer, prompts)
    expected = instance.controller(text, full).detach().numpy()
    assert (actual == expected).all()


def test_cached_states_match_real_backbone(tiny_qwen, experts, monkeypatch):
    import copy

    from teemoe.model.mixture import attach_experts

    model = copy.deepcopy(tiny_qwen)
    attach_experts(model, experts)
    tokenizer = SimpleNamespace(encode=lambda value, **kw: [int(t) for t in value.split()])
    monkeypatch.setattr(controller, "routing_view", lambda prompt: "1 2")
    cache = OrderedDict()
    text, full = controller.request_states(model, tokenizer, ["1 2 3", "1 2 4"], cache=cache)
    expected_text = controller.last_state(model, [1, 2], "cpu")
    expected_full = torch.stack([controller.last_state(model, ids, "cpu") for ids in ([1, 2, 3], [1, 2, 4])])
    assert torch.equal(text, expected_text[None].expand_as(text)) and torch.equal(full, expected_full)
