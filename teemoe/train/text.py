"""Train a text expert (native forecasting or analysis): one LoRA on the frozen backbone.

    torchrun --nproc_per_node 8 -m teemoe.train.text --config configs/native.yaml

Each update minimizes, summed over the response tokens of the global batch and
divided by their count,
    cross-entropy + kl_forward * KL(p_base || p) + kl_reverse * KL(p || p_base),
with p_base the frozen backbone: this keeps the adapted model close to it.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from ..model.backbone import BASE_MODEL, BASE_REVISION, TARGET_MODULES, decoder, load_base_model, load_tokenizer
from ..prompts import SYSTEM_PROMPT, chat_prompt, render_user
from .common import Distributed, learning_rate, load_config, log, seed_everything, setup_distributed

CHUNK = 128  # response tokens per language-model-head chunk


def tokenize(rows: list[dict], tokenizer, max_tokens: int) -> list[tuple[list[int], int]]:
    """(tokens, prompt length) per row; the response ends with EOS."""
    examples = []
    for row in rows:
        prompt = tokenizer.encode(chat_prompt(tokenizer, render_user(row), SYSTEM_PROMPT), add_special_tokens=False)
        target = tokenizer.encode(str(row["target"]), add_special_tokens=False)
        if not target or target[-1] != tokenizer.eos_token_id:
            target.append(tokenizer.eos_token_id)
        if len(prompt) + len(target) > max_tokens:
            raise ValueError(f"example of {len(prompt) + len(target)} tokens exceeds {max_tokens}")
        examples.append((prompt + target, len(prompt)))
    return examples


def plan(lengths: list[int], batch: int, bucket: int, seed: int) -> list[list[int]]:
    """One epoch of global batches: shuffle, sort each bucket of ``bucket`` batches by
    length (so a batch holds similar lengths), then shuffle the buckets."""
    rng = random.Random(seed)
    order = list(range(len(lengths)))
    rng.shuffle(order)
    full = len(order) // batch * batch
    buckets = [sorted(order[i:min(i + batch * bucket, full)], key=lambda j: (lengths[j], j))
               for i in range(0, full, batch * bucket)]
    rng.shuffle(buckets)
    order = [i for block in buckets for i in block] + order[full:]
    return [order[i:i + batch] for i in range(0, len(order), batch)]


def divergence(student: torch.Tensor, teacher: torch.Tensor, forward: float, reverse: float) -> torch.Tensor:
    """Per-token forward and reverse KL between the frozen backbone and the adapted model (log-probs)."""
    value = forward * (teacher.exp() * (teacher - student)).sum(-1)
    return value + reverse * (student.exp() * (student - teacher)).sum(-1) if reverse else value


def response_loss(student: torch.Tensor, teacher: torch.Tensor | None, labels: torch.Tensor, head,
                  kl_forward: float, kl_reverse: float, denominator: float) -> tuple[torch.Tensor, float]:
    """The objective's gradient at the final hidden states, computed through the
    language-model head a chunk at a time (the full [tokens, vocabulary] logits
    never exist at once), and the objective's sum."""
    gradient = torch.empty_like(student)
    total = 0.0
    for begin in range(0, len(labels), CHUNK):
        h = student[begin:begin + CHUNK].detach().requires_grad_(True)
        logp = head(h).float().log_softmax(-1)
        loss = torch.nn.functional.nll_loss(logp, labels[begin:begin + CHUNK], reduction="sum")
        if teacher is not None:
            with torch.no_grad():
                base = head(teacher[begin:begin + CHUNK]).float().log_softmax(-1)
            loss = loss + divergence(logp, base, kl_forward, kl_reverse).sum()
        (grad,) = torch.autograd.grad(loss / denominator, h)
        gradient[begin:begin + CHUNK] = grad
        total += float(loss.detach())
    return gradient, total


def train(config: dict, dist: Distributed) -> None:
    seed_everything(config["seed"])
    from peft import LoraConfig, get_peft_model

    tokenizer = load_tokenizer(config.get("base_model", BASE_MODEL), config.get("base_revision", BASE_REVISION))
    rows = [json.loads(line) for line in Path(config["data"]).read_text().splitlines() if line.strip()]
    examples = tokenize(rows, tokenizer, config.get("max_tokens", 32768))
    base = load_base_model(config.get("base_model", BASE_MODEL), config.get("base_revision", BASE_REVISION),
                           device=dist.device, gradient_checkpointing=True)
    lora = config["lora"]
    model = get_peft_model(base, LoraConfig(r=lora["rank"], lora_alpha=lora["alpha"],
                                            lora_dropout=lora.get("dropout", 0.0), target_modules=list(TARGET_MODULES),
                                            bias="none", task_type="CAUSAL_LM"))
    model.train()  # enables gradient checkpointing (LoRA dropout is zero)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    head = model.get_output_embeddings()
    batches = [b for epoch in range(config.get("epochs", 1))
               for b in plan([len(t) for t, _ in examples], config["global_batch_size"],
                             config.get("length_bucket_batches", 64), config["seed"] + epoch)]
    schedule = config.get("schedule_updates") or len(batches)
    output = Path(config["output"])
    start = 0
    if config.get("resume") and (output / "checkpoint" / "state.pt").exists():
        state = torch.load(output / "checkpoint" / "state.pt", map_location="cpu", weights_only=False)
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        start = state["update"]
    kl_forward, kl_reverse = float(config.get("kl_forward", 0.0)), float(config.get("kl_reverse", 0.0))
    for update in range(start, len(batches)):
        local = batches[update][dist.rank::dist.world_size]
        tokens = sum(len(examples[i][0]) - examples[i][1] for i in batches[update])
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(update, schedule, peak=config["learning_rate"],
                                        warmup_ratio=config["warmup_ratio"],
                                        final_fraction=config["final_lr_fraction"], schedule="cosine")
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for index in local:
            ids, prompt_length = examples[index]
            input_ids = torch.tensor([ids], device=dist.device)
            labels = input_ids[0, prompt_length:]
            response = slice(prompt_length - 1, -1)  # the states that predict the response tokens
            student = decoder(model)(input_ids=input_ids, use_cache=False).last_hidden_state[0, response]
            teacher = None
            if kl_forward or kl_reverse:
                with torch.no_grad(), model.disable_adapter():
                    teacher = decoder(model)(input_ids=input_ids, use_cache=False).last_hidden_state[0, response]
            gradient, value = response_loss(student, teacher, labels, head, kl_forward, kl_reverse, tokens)
            student.backward(gradient)
            loss_sum += value
        dist.sync_gradients(parameters)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        loss = dist.all_reduce(torch.tensor(loss_sum, device=dist.device)).item() / tokens
        log(dist, update=update + 1, of=len(batches), loss=loss, grad_norm=float(grad_norm),
            lr=optimizer.param_groups[0]["lr"])
        if dist.main and config.get("checkpoint_every") and (update + 1) % config["checkpoint_every"] == 0:
            save_state(model, optimizer, update + 1, output)
    if dist.main:
        model.save_pretrained(output / "adapter")
        print(f"saved adapter to {output / 'adapter'}", flush=True)
    dist.barrier()


def save_state(model, optimizer, update: int, output: Path) -> None:
    from peft import get_peft_model_state_dict

    path = output / "checkpoint"
    path.mkdir(parents=True, exist_ok=True)
    torch.save(dict(adapter=get_peft_model_state_dict(model), optimizer=optimizer.state_dict(), update=update),
               path / "state.tmp")
    (path / "state.tmp").replace(path / "state.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data")
    parser.add_argument("--output")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, dict(data=args.data, output=args.output, learning_rate=args.learning_rate,
                                           resume=args.resume or None))
    train(config, setup_distributed())


if __name__ == "__main__":
    main()
