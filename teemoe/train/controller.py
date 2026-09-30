"""Train the controller that weights the three frozen experts for each request.

    torchrun --nproc_per_node 8 -m teemoe.train.controller --config configs/controller.yaml

One epoch over 334 aggregation, 333 native-forecasting and 333 analysis
examples drawn from the experts' own training data. Every update takes eight
examples of each capability; each capability's loss is normalized separately
and the three are averaged:

  text (native, analysis)  cross-entropy of the response under the routed mixture
                           - log(1 - w_aggregation)
  aggregation              the editor's forecast loss under the routed mixture
                           - log(w_aggregation)

Only the controller head and its input balance are trained.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from ..model.backbone import BASE_MODEL, BASE_REVISION, decoder, load_base_model, load_tokenizer
from ..model.controller import AGGREGATION, Controller, last_state
from ..model.editor import AggregationEditor, editor_loss
from ..model.mixture import attach_experts, expert_weights
from ..prompts import SYSTEM_PROMPT, chat_prompt, forecast_message, history_timestamps, render_user, routing_view
from .common import Distributed, load_config, log, seed_everything, setup_distributed
from .text import response_loss

BUCKET = 64  # length-sorted text examples come from a random bucket of 64 batches


def text_order(lengths: list[int], count: int, batch: int, rng: random.Random) -> list[int]:
    """``count`` examples: a random bucket of ``BUCKET`` batches, shortest first."""
    bucket = rng.sample(range(len(lengths)), min(len(lengths), BUCKET * batch))
    return sorted(bucket, key=lambda i: (lengths[i], i))[:count]


def forecast_prompt(tokenizer, request: dict) -> str:
    """The chat prompt of a numerical forecasting request (``history``, ``start``, ``frequency``, ``horizon``)."""
    past, future = history_timestamps(request["start"], request["frequency"], len(request["history"]),
                                      int(request["horizon"]))
    return chat_prompt(tokenizer, forecast_message(request["history"], past, future))


def training_examples(tokenizer, data: dict, counts: dict, batch: int, seed: int):
    """The controller's examples: for each text capability, ``(prompt, prompt ids, target ids)``
    in training order; for aggregation, editor-cache rows; and the cache's requests."""
    rng, encode, text = random.Random(seed), (lambda t: tokenizer.encode(t, add_special_tokens=False)), {}
    for name in ("native", "analysis"):
        examples = []
        for line in Path(data[name]).open():
            if line.strip():
                row = json.loads(line)
                prompt = chat_prompt(tokenizer, render_user(row), SYSTEM_PROMPT)
                target = encode(str(row["target"]))
                target += [] if target and target[-1] == tokenizer.eos_token_id else [tokenizer.eos_token_id]
                examples.append((prompt, encode(prompt), target))
        order = text_order([len(p) + len(t) for _, p, t in examples], counts[name], batch, rng)
        text[name] = [examples[i] for i in order]
    requests = [json.loads(line) for line in (Path(data["aggregation"]) / "requests.jsonl").open()]
    return text, rng.sample(range(len(requests)), counts["aggregation"]), requests


def train(config: dict, dist: Distributed) -> None:
    from safetensors.torch import load_file, save_file

    seed_everything(config.get("seed", 1))
    batch = int(config.get("batch_per_capability", 8))
    counts = config.get("examples", dict(aggregation=334, native=333, analysis=333))
    tokenizer = load_tokenizer(config.get("base_model", BASE_MODEL), config.get("base_revision", BASE_REVISION))
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)

    # ----------------------------------------------------------------- data
    text, numeric, requests = training_examples(tokenizer, config["data"], counts, batch, config.get("seed", 1))
    cache = Path(config["data"]["aggregation"])
    arrays = {name: np.load(cache / f"{name}.npy", mmap_mode="r")
              for name in ("candidates", "parent", "target", "knot_weight")}

    # ---------------------------------------------------------------- model
    model = load_base_model(config.get("base_model", BASE_MODEL), config.get("base_revision", BASE_REVISION),
                            device=dist.device, gradient_checkpointing=True)
    model.train()  # gradient checkpointing only; nothing in the backbone is trained
    experts = config["experts"]
    attach_experts(model, [experts["aggregation"], experts["native"], experts["analysis"]])
    hidden = model.config.get_text_config().hidden_size
    editor = AggregationEditor(hidden).to(dist.device).requires_grad_(False)
    editor.load_state_dict(load_file(experts["editor"]))
    controller = Controller(hidden).to(dist.device)
    optimizer = torch.optim.AdamW([dict(params=controller.head.parameters(), lr=config.get("learning_rate", 1e-4)),
                                   dict(params=[controller.signal_fraction],
                                        lr=config.get("balance_learning_rate", 0.05))],
                                  weight_decay=config.get("weight_decay", 0.0))
    parameters = list(controller.parameters())
    head = model.get_output_embeddings()

    def route(prompt: str) -> torch.Tensor:
        """Expert weights [1, 3] from the frozen backbone's task-text and full-request states."""
        states = [last_state(model, encode(view), dist.device) for view in (routing_view(prompt), prompt)]
        return controller(states[0][None], states[1][None])

    def text_step(examples, rank_rows):
        tokens = sum(len(target) for _, _, target in examples)
        total = 0.0
        for prompt, prompt_ids, target in rank_rows:
            weights = route(prompt)
            start = len(prompt_ids)
            input_ids = torch.tensor([prompt_ids + target], device=dist.device)
            with expert_weights(model, weights):
                states = decoder(model)(input_ids=input_ids, use_cache=False).last_hidden_state[0, start - 1:-1]
                gradient, value = response_loss(states, None, input_ids[0, start:], head, 0.0, 0.0,
                                                3 * tokens)
                routing_loss = -torch.log((1 - weights[0, AGGREGATION]).clamp_min(1e-7)) / (3 * len(examples))
                torch.autograd.backward((states, routing_loss), (gradient, None))
            total += value / (3 * tokens) + float(routing_loss.detach())
        return total

    def numeric_step(rows, rank_rows):
        total = 0.0
        for row in rank_rows:
            weights = route(forecast_prompt(tokenizer, requests[row]))
            tensor = lambda name: torch.as_tensor(np.asarray(arrays[name][[row]]), device=dist.device)
            with expert_weights(model, weights):
                _, correction = editor(decoder(model), tensor("candidates"), tensor("parent"))
                loss = editor_loss(correction, target=tensor("target"), reference=tensor("parent"),
                                   candidates=tensor("candidates"), knot_weight=tensor("knot_weight"),
                                   beta=config.get("beta", 0.05), edit_weight=0.0).total / (3 * len(rows))
                loss = loss - torch.log(weights[0, AGGREGATION].clamp_min(1e-7)) / (3 * len(rows))
                loss.backward()
            total += float(loss.detach())
        return total

    # ----------------------------------------------------------------- loop
    updates = max(-(-len(v) // batch) for v in (text["native"], text["analysis"], numeric))
    for update in range(updates):
        optimizer.zero_grad(set_to_none=True)
        loss = 0.0
        for name in ("native", "analysis"):
            examples = text[name][update * batch:(update + 1) * batch]
            loss += text_step(examples, examples[dist.rank::dist.world_size]) if examples else 0.0
        rows = numeric[update * batch:(update + 1) * batch]
        loss += numeric_step(rows, rows[dist.rank::dist.world_size]) if rows else 0.0
        dist.sync_gradients(parameters)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        with torch.no_grad():
            controller.signal_fraction.clamp_(0, 1)
        loss = dist.all_reduce(torch.tensor(loss, device=dist.device)).item()
        log(dist, update=update + 1, of=updates, loss=loss, grad_norm=float(grad_norm),
            signal_fraction=float(controller.signal_fraction))
    if dist.main:
        output = Path(config["output"])
        output.mkdir(parents=True, exist_ok=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in controller.state_dict().items()},
                  str(output / "controller.safetensors"))
        print(f"saved controller to {output / 'controller.safetensors'}", flush=True)
    dist.barrier()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    train(load_config(args.config, dict(output=args.output)), setup_distributed())


if __name__ == "__main__":
    main()
