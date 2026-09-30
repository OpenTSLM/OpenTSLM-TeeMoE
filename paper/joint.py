"""The joint baseline: one shared LoRA and numerical port trained on all three datasets,
then a binary selector between numerical and text output.

    torchrun --nproc_per_node 8 paper/joint.py train --output artifacts/joint
    torchrun --nproc_per_node 8 paper/joint.py select --output artifacts/joint
    python paper/joint.py package --output artifacts/joint     # -> artifacts/joint/checkpoint
    python paper/ablations.py joint gift --checkpoint artifacts/joint/checkpoint

Training uses the experts' data (20,000 native, 12,000 analysis and 4,096 aggregation
presentations), shuffled together into 4,512 updates of 8 examples; each capability's
mean loss is weighted by its share of the batch. The selector sees the same routing
inputs as TeeMoE's controller and is trained on the controller's 1,000 examples.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from teemoe.model.backbone import (BASE_MODEL, BASE_REVISION, TARGET_MODULES, decoder,  # noqa: E402
                                   load_base_model, load_tokenizer)
from teemoe.model.controller import last_state, request_states  # noqa: E402
from teemoe.model.editor import AggregationEditor, editor_loss  # noqa: E402
from teemoe.model.mixture import attach_experts  # noqa: E402
from teemoe.prompts import routing_view  # noqa: E402
from teemoe.teemoe import TeeMoE  # noqa: E402
from teemoe.train.aggregation import sample_plan  # noqa: E402
from teemoe.train.common import learning_rate, log, seed_everything, setup_distributed  # noqa: E402
from teemoe.train.controller import forecast_prompt, training_examples  # noqa: E402
from teemoe.train.text import response_loss, tokenize  # noqa: E402

DATA = dict(native="data/native/train.jsonl", analysis="data/analysis/train.jsonl",
            aggregation="artifacts/aggregation/editor_cache")


def load_cache(path: Path) -> dict[str, np.ndarray]:
    return {n: np.load(path / f"{n}.npy") for n in ("candidates", "parent", "target", "knot_weight", "group")}


# ------------------------------------------------------------------ training
def train(args, dist) -> None:
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file

    seed_everything(1)
    tokenizer = load_tokenizer()
    text = {name: tokenize([json.loads(line) for line in open(DATA[name]) if line.strip()], tokenizer, 32768)
            for name in ("native", "analysis")}
    cache = load_cache(Path(DATA["aggregation"]))
    numeric = np.concatenate(sample_plan(cache["group"], updates=128, batch=32, seed=1))
    items = [("native", i) for i in range(len(text["native"]))] + \
            [("analysis", i) for i in range(len(text["analysis"]))] + [("aggregation", int(j)) for j in numeric]
    random.Random(20260901).shuffle(items)
    batches = [items[i:i + 8] for i in range(0, len(items), 8)]
    base = load_base_model(BASE_MODEL, BASE_REVISION, device=dist.device, gradient_checkpointing=True)
    model = get_peft_model(base, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, target_modules=list(TARGET_MODULES),
                                            bias="none", task_type="CAUSAL_LM"))
    model.train()
    editor = AggregationEditor(base.config.get_text_config().hidden_size).to(dist.device)
    lora = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora + list(editor.parameters()), lr=1e-5, weight_decay=0.003)
    head = model.get_output_embeddings()
    tensor = lambda name, rows: torch.as_tensor(cache[name][rows], device=dist.device)
    for update, batch in enumerate(batches):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(update, len(batches), peak=1e-5, warmup_ratio=0.05, final_fraction=0.1)
        optimizer.zero_grad(set_to_none=True)  # the numerical port only moves when it sees an example
        total = 0.0
        for capability in ("native", "analysis", "aggregation"):
            members = [i for kind, i in batch if kind == capability]
            if not members:
                continue
            share = len(members) / len(batch)
            local = members[dist.rank::dist.world_size]
            if capability == "aggregation":
                if local:
                    _, correction = editor(decoder(model), tensor("candidates", local), tensor("parent", local))
                    loss = editor_loss(correction, target=tensor("target", local), reference=tensor("parent", local),
                                       candidates=tensor("candidates", local),
                                       knot_weight=tensor("knot_weight", local)).total * share / len(members)
                    loss.backward()
                    total += float(loss)
                continue
            examples = text[capability]
            tokens = sum(len(examples[i][0]) - examples[i][1] for i in members)
            for i in local:
                ids, start = examples[i]
                input_ids = torch.tensor([ids], device=dist.device)
                states = decoder(model)(input_ids=input_ids, use_cache=False).last_hidden_state[0, start - 1:-1]
                with torch.no_grad(), model.disable_adapter():
                    teacher = decoder(model)(input_ids=input_ids, use_cache=False).last_hidden_state[0, start - 1:-1]
                gradient, value = response_loss(states, teacher, input_ids[0, start:], head, 0.5, 0.05,
                                                tokens / share)
                states.backward(gradient)
                total += value * share / tokens
        moving = lora + (list(editor.parameters()) if any(kind == "aggregation" for kind, _ in batch) else [])
        dist.sync_gradients(moving)
        grad_norm = torch.nn.utils.clip_grad_norm_(moving, 1.0)
        optimizer.step()
        loss = dist.all_reduce(torch.tensor(total, device=dist.device)).item()
        log(dist, update=update + 1, of=len(batches), loss=loss, grad_norm=float(grad_norm))
    if dist.main:
        model.save_pretrained(Path(args.output) / "adapter")
        save_file({k: v.detach().cpu().contiguous() for k, v in editor.state_dict().items()},
                  str(Path(args.output) / "editor.safetensors"))
    dist.barrier()


# ------------------------------------------------------------------ selector
class Selector(torch.nn.Module):
    """P(numerical output) = sigmoid(w . h + b), h mixing task-text and full-request states."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.head = torch.nn.Linear(hidden_size, 1)
        self.signal_fraction = torch.nn.Parameter(torch.zeros(()))
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.constant_(self.head.bias, math.log(0.5))  # starts at P = 1/3

    def forward(self, text_state: torch.Tensor, full_state: torch.Tensor) -> torch.Tensor:
        mixed = (1 - self.signal_fraction) * text_state.float() + self.signal_fraction * full_state.float()
        return torch.sigmoid(self.head(mixed)[..., 0])


def select(args, dist) -> None:
    from safetensors.torch import save_file

    seed_everything(1)
    tokenizer = load_tokenizer()
    text, numeric, requests = training_examples(tokenizer, DATA, dict(aggregation=334, native=333, analysis=333),
                                                8, 1)  # the controller's examples
    prompts = {name: [prompt for prompt, _, _ in text[name]] for name in text}
    prompts["aggregation"] = [forecast_prompt(tokenizer, requests[i]) for i in numeric]
    model = load_base_model(BASE_MODEL, BASE_REVISION, device=dist.device)
    attach_experts(model, [])
    selector = Selector(model.config.get_text_config().hidden_size).to(dist.device)
    optimizer = torch.optim.AdamW([dict(params=selector.head.parameters(), lr=1e-4),
                                   dict(params=[selector.signal_fraction], lr=0.05)], weight_decay=0.0)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    updates = max(-(-len(v) // 8) for v in prompts.values())
    for update in range(updates):
        optimizer.zero_grad(set_to_none=True)
        for name, chats in prompts.items():
            batch = chats[update * 8:(update + 1) * 8]
            for prompt in batch[dist.rank::dist.world_size]:
                states = [last_state(model, encode(view), dist.device)[None] for view in (routing_view(prompt), prompt)]
                probability = selector(*states)[0].clamp(1e-7, 1 - 1e-7)
                target = 1.0 if name == "aggregation" else 0.0
                loss = -(target * torch.log(probability) + (1 - target) * torch.log(1 - probability))
                (loss / len(batch) / 3).backward()
        dist.sync_gradients(list(selector.parameters()))
        torch.nn.utils.clip_grad_norm_(list(selector.parameters()), 1.0)
        optimizer.step()
        with torch.no_grad():
            selector.signal_fraction.clamp_(0, 1)
        log(dist, update=update + 1, of=updates)
    if dist.main:
        save_file({k: v.detach().cpu().contiguous() for k, v in selector.state_dict().items()},
                  str(Path(args.output) / "selector.safetensors"))
    dist.barrier()


# ------------------------------------------------------------------- runtime
def package(args) -> Path:
    import shutil

    output, aggregation = Path(args.output), Path(args.aggregation)
    target = output / "checkpoint"
    (target / "adapters").mkdir(parents=True, exist_ok=True)
    shutil.copytree(output / "adapter", target / "adapters/joint", dirs_exist_ok=True)
    for name in ("editor.safetensors", "selector.safetensors"):
        shutil.copyfile(output / name, target / name)
    shutil.copyfile(aggregation / "routers/full.ubj", target / "router.ubj")
    fnf_mass = json.loads((aggregation / "blend.json").read_text())["fnf_mass"]
    (target / "teemoe.json").write_text(json.dumps(dict(base_model=BASE_MODEL, base_revision=BASE_REVISION,
                                                        experts=["joint"], fnf_mass=fnf_mass), indent=2) + "\n")
    return target


class JointTeeMoE(TeeMoE):
    """The joint baseline behind TeeMoE's interface: one adapter, a binary output selector."""

    def __init__(self, root, *, device="cuda:0", backend="vllm", vllm_devices=None, vllm_python=None,
                 forecast_devices=None, environments=None, granite=None, fnf_root=None) -> None:
        from safetensors.torch import load_file

        self.root, self.device, self.backend = Path(root), torch.device(device), backend
        self.config = json.loads((self.root / "teemoe.json").read_text())
        self.vllm_devices, self.vllm_python = vllm_devices, vllm_python
        self.forecast_devices = forecast_devices or (str(self.device),)
        self.environments, self.granite, self.fnf_root, self._ensemble = environments or {}, granite, fnf_root, None
        self.tokenizer = load_tokenizer(self.config["base_model"], self.config["base_revision"])
        self.model = load_base_model(self.config["base_model"], self.config["base_revision"], device=self.device)
        self.adapters = [self.root / "adapters/joint"]
        attach_experts(self.model, self.adapters)
        hidden = self.model.config.get_text_config().hidden_size
        self.selector = Selector(hidden).to(self.device)
        self.selector.load_state_dict(load_file(str(self.root / "selector.safetensors")))
        self.editor = AggregationEditor(hidden).to(self.device)
        self.editor.load_state_dict(load_file(str(self.root / "editor.safetensors")))

    @torch.no_grad()
    def route(self, chat_prompts):
        p = self.selector(*request_states(self.model, self.tokenizer, chat_prompts)).cpu().numpy()
        return np.stack([p, 1 - p, np.zeros_like(p)], axis=1)  # reported as (numerical, text, -)

    def execution_weights(self, weights):
        return np.ones((len(weights), 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("train", "select", "package"))
    parser.add_argument("--output", default="artifacts/joint")
    parser.add_argument("--aggregation", default="artifacts/aggregation", help="(package) the ensemble to use")
    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    if args.stage == "package":
        print(package(args))
    else:
        (train if args.stage == "train" else select)(args, setup_distributed())


if __name__ == "__main__":
    main()
