import copy
import re

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from teemoe.generation import _regex_constraint, generate_transformers
from teemoe.model.mixture import attach_experts
from teemoe.prompts import parse_forecast


@pytest.fixture
def tokenizer():
    characters = sorted(set("0123456789abcdeforstx<>/() ,.:\n-+"))
    vocab = {token: i for i, token in enumerate(["<unk>", "<eos>"] + characters)}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    backend.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                  eos_token="<eos>", pad_token="<eos>")


def test_regex_constraint_filters_tokens_and_finishes(tokenizer):
    constraint = _regex_constraint(tokenizer, "ab")
    ids = tokenizer.encode("x", add_special_tokens=False)
    for character in "ab":
        token = tokenizer.convert_tokens_to_ids(character)
        assert constraint(0, torch.tensor(ids)) == [token]
        assert constraint(1, torch.tensor(ids)) == [token]
        ids.append(token)
    assert constraint(0, torch.tensor(ids)) == [tokenizer.eos_token_id]


def test_regex_constraint_reuses_vocabulary_not_parser_state(tokenizer):
    first = _regex_constraint(tokenizer, "a")
    cached = tokenizer._teemoe_enforcer_data
    second = _regex_constraint(tokenizer, "b")
    assert tokenizer._teemoe_enforcer_data is cached
    ids = torch.tensor(tokenizer.encode("x", add_special_tokens=False))
    assert first(0, ids) == [tokenizer.convert_tokens_to_ids("a")]
    assert second(0, ids) == [tokenizer.convert_tokens_to_ids("b")]


@pytest.mark.parametrize("samples", [2, 5])
def test_transformers_constrained_forecast_generation(tiny_qwen, experts, tokenizer, samples):
    model = copy.deepcopy(tiny_qwen)
    attach_experts(model, experts)
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    expected = "<forecast>\n(2024-01-01 00:00:00, 7)\n</forecast>"
    output = generate_transformers(model, tokenizer, [dict(
        prompt="x", weights=[0.2, 0.5, 0.3], samples=samples, temperature=1.0,
        max_tokens=len(expected) + 2, regex=re.escape(expected), seed=1)])
    assert output == [[expected] * samples]
    assert parse_forecast(output[0][0], ["2024-01-01 00:00:00"]) == [7.0]


def test_transformers_sample_batches_preserve_count(tiny_qwen, experts, tokenizer, monkeypatch):
    model = copy.deepcopy(tiny_qwen)
    attach_experts(model, experts)
    batches = []

    def generate(input_ids, num_return_sequences, **kwargs):
        batches.append(num_return_sequences)
        suffix = torch.tensor([[tokenizer.convert_tokens_to_ids("a")]])
        return torch.cat((input_ids, suffix), dim=1).repeat(num_return_sequences, 1)

    monkeypatch.setattr(model, "generate", generate)
    request = dict(prompt="x", weights=[0.2, 0.5, 0.3], samples=9, temperature=1.0, max_tokens=1)
    assert generate_transformers(model, tokenizer, [request]) == [["a"] * 9]
    assert batches == [4, 4, 1]
    with pytest.raises(ValueError, match="samples must be positive"):
        generate_transformers(model, tokenizer, [dict(request, samples=0)])
