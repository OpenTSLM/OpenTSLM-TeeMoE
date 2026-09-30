"""The learned controller that weights the three experts for each request."""

from __future__ import annotations

from collections import OrderedDict

import torch

from ..prompts import routing_view
from .backbone import decoder
from .mixture import expert_weights

AGGREGATION = 0  # expert order: aggregation, native forecasting, analysis
NUMERICAL_THRESHOLD = 0.5  # aggregation weight above which the numerical decoder answers
STATE_CACHE_SIZE = 256


class Controller(torch.nn.Module):
    """softmax(W h + b) with h = (1 - a) * task_text_state + a * full_request_state."""

    def __init__(self, hidden_size: int, experts: int = 3) -> None:
        super().__init__()
        self.head = torch.nn.Linear(hidden_size, experts)
        self.signal_fraction = torch.nn.Parameter(torch.zeros(()))
        torch.nn.init.zeros_(self.head.weight)  # start from a uniform mixture
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, text_state: torch.Tensor, full_state: torch.Tensor) -> torch.Tensor:
        alpha = self.signal_fraction
        mixed = (1 - alpha) * text_state.float() + alpha * full_state.float()
        return torch.softmax(self.head(mixed), dim=-1)


@torch.no_grad()
def last_state(model, input_ids: list[int], device) -> torch.Tensor:
    """Final hidden state of the last prompt token, with every expert disabled."""
    ids = torch.tensor([input_ids], device=device)
    with expert_weights(model, None):
        output = decoder(model)(input_ids=ids, attention_mask=torch.ones_like(ids),
                                position_ids=torch.arange(ids.shape[1], device=device)[None],
                                use_cache=False, return_dict=True)
    return output.last_hidden_state[0, -1].float()


@torch.no_grad()
def request_states(model, tokenizer, prompts: list[str], *, text: bool = True,
                   full: bool = True, cache: OrderedDict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Task-text and full-request states for chat-formatted prompts ([n, hidden] each); a view
    that is not requested is returned as zeros."""
    device = next(model.parameters()).device
    encode = lambda value: tokenizer.encode(value, add_special_tokens=False)
    cache = OrderedDict() if cache is None else cache

    def state(prompt):
        key = tuple(encode(prompt))
        if key not in cache:
            cache[key] = last_state(model, list(key), device).detach().cpu()
            if len(cache) > STATE_CACHE_SIZE:
                cache.popitem(last=False)
        cache.move_to_end(key)
        return cache[key]

    views = []
    for wanted, view in ((text, routing_view), (full, lambda prompt: prompt)):
        states = [state(view(p)) for p in prompts] if wanted else None
        views.append(states)
    width = next(len(v[0]) for v in views if v is not None)
    return tuple(torch.stack(v).to(device) if v is not None else torch.zeros(len(prompts), width, device=device)
                 for v in views)
