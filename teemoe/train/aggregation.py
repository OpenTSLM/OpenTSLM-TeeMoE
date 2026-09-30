"""Train the numerical ensemble and the aggregation expert on the training windows.

    python -m teemoe.train.aggregation prepare --config configs/aggregation.yaml
    torchrun --nproc_per_node 2 -m teemoe.train.aggregation editor --config configs/aggregation.yaml

``prepare`` runs these stages, reusing every finished one:

  forecasts  the 8 core forecasters and the 10 FnF members on every window
  table      router inputs, CRPS ranks, candidates and targets at 8 horizon knots
  routers    the XGBoost router on all windows, and on 10 folds that each hold out
             one tenth of the physical series
  parents    the FnF blend weight, and out-of-fold reference forecasts for the
             editor's training windows (GIFT-Eval training split)

``editor`` then trains the rank-4 LoRA with the numerical connector and decoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from ..data.aggregation import BLOCKS
from ..ensemble.fnf import FNF_MEMBERS, TotoFnF, blend, canonical_frequency, ensure_fnf, fit_blend_mass
from ..ensemble.forecasters import (CORE_MODELS, EXTRA_CANDIDATES, FNF_MODELS, knots, load, output_name,
                                    run_all)
from ..ensemble.router import (CORE, Router, RouterRecipe, crps_ranks, fit_router, frequency_seasonality,
                               pool_quantiles, router_features)
from .common import Distributed, load_config, log, seed_everything, setup_distributed

FOLDS = 10
KNOTS = 8
DEFAULT_START = "2000-01-03"
ROUTING_HISTORY = 168  # observations shown in a forecasting prompt


def say(**values) -> None:
    print(json.dumps(values), flush=True)


def devices(config: dict) -> list[str]:
    """Configured devices: GPU indices, or device names such as "cpu"."""
    return [d if isinstance(d, str) else f"cuda:{d}" for d in config.get("devices", [0])]


def windows(data: Path):
    """Every training window, in block order."""
    manifest = json.loads((data / "manifest.json").read_text())
    for block in BLOCKS:
        with (data / manifest["blocks"][block]["path"]).open() as stream:
            for line in stream:
                yield json.loads(line)


# ------------------------------------------------------------------ forecasts
def stage_forecasts(config: dict, output: Path) -> Path:
    requests = output / "requests.jsonl"
    if not requests.exists():
        temporary = requests.with_suffix(".tmp")
        with temporary.open("w") as stream:
            for w in windows(Path(config["data"])):
                stream.write(json.dumps(dict(
                    history=w["history"], horizon=len(w["future"]), frequency=w["freq"],
                    start=w.get("history_start") or DEFAULT_START, domain=w.get("domain"), term=w["term"],
                    dataset=w["dataset"])) + "\n")
        temporary.rename(requests)
    gpus = devices(config)
    return run_all(requests, output / "forecasts", devices=gpus, environments=config.get("environments"),
                   granite=config.get("granite"), shards=int(config.get("forecast_shards", len(gpus))),
                   at_knots=[f"fnf/{m}" for m in FNF_MODELS])


# ---------------------------------------------------------------------- table
_FNF: TotoFnF | None = None
_EDITOR_BLOCKS: tuple[str, ...] = ()


def _init_worker(fnf_root: str, editor_blocks: tuple[str, ...]) -> None:
    global _FNF, _EDITOR_BLOCKS
    _FNF, _EDITOR_BLOCKS = TotoFnF(fnf_root), editor_blocks


def _valid(q: np.ndarray) -> np.ndarray:
    return np.isfinite(q).all(axis=(-2, -1)) & (np.abs(q) < 1e30).all(axis=(-2, -1))


def table_row(item):
    """Router inputs, ranks and knot tensors for one window (``keep`` is False if a core forecast failed)."""
    window, core, members = item
    core = np.stack(core)
    identity = dict(group=int.from_bytes(hashlib.sha256(window["source_group"].encode()).digest()[:8], "little"),
                    block=BLOCKS.index(window["block"]), keep=False)
    if not _valid(core).all():
        return identity
    history = np.asarray(window["history"], np.float64)
    future = np.asarray(window["future"], np.float64)
    horizon, frequency = len(future), canonical_frequency(window["freq"])
    magnitude = max(float(np.max(np.abs(history))), 1.0)
    knot = knots(horizon)
    nearest = np.argmin(np.abs(np.arange(horizon)[:, None] - knot[None, :]), axis=1)
    season = max(frequency_seasonality(frequency), 1)
    lag = season if len(history) > season else 1
    seasonal = float(np.mean(np.abs(history[lag:] - history[:-lag])) / magnitude) if len(history) > lag else np.nan
    mase_valid = bool(np.isfinite(seasonal) and seasonal > 0)
    core_knots = (core.astype(np.float64) / magnitude)[:, knot].astype(np.float32)
    target = (future / magnitude)[knot].astype(np.float32)
    members = np.stack(members).astype(np.float64) / magnitude  # [10, 8, 9]
    record = None
    if _valid(members).all() and _FNF.supports(frequency, window["term"]):
        record = _FNF.feature_record(history, frequency=frequency, horizon=horizon, domain=window.get("domain"),
                                     num_variates=int(window.get("num_variates", 1)))
    return dict(
        identity, keep=True, routing=routing_request(window) if window["block"] in _EDITOR_BLOCKS else None,
        selected=router_features(history, core, frequency=frequency, term=window["term"],
                                 history_context=len(history)),
        ranks=crps_ranks(core_knots[None], target[None], np.float32([seasonal if mase_valid else 1.0]))[0],
        core=core_knots, members=members.astype(np.float32), target=target,
        knot_weight=np.bincount(nearest, minlength=KNOTS).astype(np.float32), magnitude=np.float32(magnitude),
        mase_valid=mase_valid, record=record, bucket=f"{frequency.split('-', 1)[0]}|{window['term']}",
        label=f"{window['freq']}|{horizon}")


def routing_request(window: dict) -> dict:
    """The forecasting request a window poses (its last observations, for routing prompts)."""
    import pandas as pd

    history = window["history"][-ROUTING_HISTORY:]
    start = pd.Period(window.get("history_start") or DEFAULT_START, freq=window["freq"]) \
        + (len(window["history"]) - len(history))
    return dict(history=history, start=str(start.start_time), frequency=window["freq"],
                horizon=len(window["future"]))


def stage_table(config: dict, output: Path) -> Path:
    table = output / "table"
    if (table / "done").exists():
        return table
    fnf_root = str(config.get("fnf_root") or ensure_fnf())
    forecasts = output / "forecasts"
    core = [load(forecasts / output_name(f"core/{m}")) for m in CORE_MODELS]
    members = [load(forecasts / output_name(f"fnf/{m}")) for m in FNF_MEMBERS]
    rows = len(core[0])
    table.mkdir(parents=True, exist_ok=True)
    shapes = dict(selected=((rows, 195), np.float32), ranks=((rows, CORE), np.uint8),
                  core=((rows, CORE, KNOTS, 9), np.float32), members=((rows, 10, KNOTS, 9), np.float32),
                  target=((rows, KNOTS), np.float32), knot_weight=((rows, KNOTS), np.float32),
                  magnitude=((rows,), np.float32), mase_valid=((rows,), bool), keep=((rows,), bool),
                  group=((rows,), np.uint64), block=((rows,), np.int8))
    arrays = {name: np.lib.format.open_memmap(table / f"{name}.npy", mode="w+", shape=shape, dtype=dtype)
              for name, (shape, dtype) in shapes.items()}
    labels, buckets, records = [""] * rows, [""] * rows, {}
    editor_blocks = tuple(config.get("editor_blocks", ("gift_sampled", "gift_rolling")))
    routing = (table / "routing.jsonl").open("w")
    source, index = windows(Path(config["data"])), 0
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(int(config.get("workers", 16)), mp_context=context, initializer=_init_worker,
                             initargs=(fnf_root, editor_blocks)) as pool:
        while chunk := [w for _, w in zip(range(4096), source)]:  # bounded memory
            items = [(w, [c[index + j] for c in core], [m[index + j] for m in members])
                     for j, w in enumerate(chunk)]
            for i, result in enumerate(pool.map(table_row, items, chunksize=32), start=index):
                arrays["group"][i], arrays["block"][i], arrays["keep"][i] = (result["group"], result["block"],
                                                                             result["keep"])
                if not result["keep"]:
                    continue
                for name in ("selected", "ranks", "core", "members", "target", "knot_weight", "magnitude",
                             "mase_valid"):
                    arrays[name][i] = result[name]
                labels[i], buckets[i] = result["label"], result["bucket"]
                if result["record"] is not None:
                    records[i] = result["record"]
                if result["routing"] is not None:
                    routing.write(json.dumps(dict(index=i, **result["routing"])) + "\n")
            index += len(chunk)
            say(stage="table", rows=index, of=rows)
    routing.close()
    # FnF: gate the ten members per frequency/term bucket (in bulk).
    fnf, gated = TotoFnF(fnf_root), np.full((rows, KNOTS, 9), np.nan, np.float32)
    by_bucket: dict[str, list[int]] = {}
    for i in records:
        by_bucket.setdefault(buckets[i], []).append(i)
    for bucket, indices in by_bucket.items():
        frequency, term = bucket.split("|")
        weights = fnf.weights(fnf.frame([records[i] for i in indices]), frequency=frequency, term=term)
        combined = np.einsum("rm,rmkq->rkq", weights, arrays["members"][indices].astype(np.float64))
        gated[indices] = np.maximum.accumulate(combined, axis=-1)
    np.save(table / "fnf.npy", gated)
    np.save(table / "labels.npy", np.asarray(labels))
    for array in arrays.values():
        array.flush()
    (table / "done").write_text(json.dumps(dict(rows=rows, kept=int(arrays["keep"].sum()),
                                                fnf_rows=len(records))) + "\n")
    return table


def read_table(output: Path) -> dict[str, np.ndarray]:
    table = output / "table"
    names = ("selected", "ranks", "core", "members", "target", "knot_weight", "magnitude", "mase_valid",
             "keep", "group", "block", "fnf", "labels")
    return {name: np.load(table / f"{name}.npy", mmap_mode="r") for name in names}


def folds(table: dict) -> np.ndarray:
    return (np.asarray(table["group"]) % np.uint64(FOLDS)).astype(np.int8)


# -------------------------------------------------------------------- routers
def _fit(task) -> str:
    output, name, device, recipe = task
    table = read_table(Path(output))
    keep = np.asarray(table["keep"])
    rows = np.flatnonzero(keep if name == "full" else keep & (folds(table) != int(name[-2:])))
    path = Path(output) / "routers" / f"{name}.ubj"
    if not path.exists():
        fit_router(np.asarray(table["selected"][rows]), np.asarray(table["ranks"][rows]),
                   path.with_suffix(".tmp"), RouterRecipe(**recipe), device=device)
        path.with_suffix(".tmp").rename(path)
    return name


def _pool(task) -> str:
    """The router's pooled forecast at the knots for every kept window (normalized units)."""
    output, name, device, recipe = task
    path = Path(output) / "parents" / f"left_{name}.npy"
    if path.exists():
        return name
    table = read_table(Path(output))
    router = Router(Path(output) / "routers" / f"{name}.ubj", RouterRecipe(**recipe), device=device)
    rows = len(table["keep"])
    left = np.full((rows, KNOTS, 9), np.nan, np.float32)
    kept = np.flatnonzero(table["keep"])
    for begin in range(0, len(kept), 65536):
        chunk = kept[begin:begin + 65536]
        weights = router.allocations(np.asarray(table["selected"][chunk]))
        left[chunk] = pool_quantiles(np.asarray(table["core"][chunk]), weights)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix(".tmp.npy"), left)
    path.with_suffix(".tmp.npy").rename(path)
    return name


def in_waves(function, output: Path, config: dict) -> None:
    """Run ``function`` for the full router and the ten fold routers, one per device at a time."""
    gpus = devices(config)
    names = ["full"] + [f"fold{k:02d}" for k in range(FOLDS)]
    recipe = config.get("router", {})
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(len(gpus), mp_context=context) as pool:
        for begin in range(0, len(names), len(gpus)):
            wave = [(str(output), name, device, recipe) for name, device in zip(names[begin:], gpus)]
            for name in pool.map(function, wave):
                say(stage=function.__name__.strip("_"), router=name, status="complete")


# -------------------------------------------------------------------- parents
def stage_parents(config: dict, output: Path) -> Path:
    cache = output / "editor_cache"
    if (cache / "done").exists():
        return cache
    in_waves(_pool, output, config)
    table = read_table(output)
    keep, fold = np.asarray(table["keep"]), folds(table)
    fnf = np.asarray(table["fnf"])
    usable = keep & np.asarray(table["mase_valid"]) & np.isfinite(fnf).all(axis=(1, 2))

    def mass(left, rows):
        return fit_blend_mass(left[rows, :, 4], fnf[rows, :, 4], np.asarray(table["target"][rows]),
                              np.asarray(table["knot_weight"][rows]))

    full = np.load(output / "parents" / "left_full.npy", mmap_mode="r")
    fnf_mass = mass(full, np.flatnonzero(usable))
    (output / "blend.json").write_text(json.dumps(dict(fnf_mass=fnf_mass), indent=2) + "\n")
    say(stage="parents", fnf_mass=fnf_mass)
    editor_blocks = [BLOCKS.index(b) for b in config.get("editor_blocks", ("gift_sampled", "gift_rolling"))]
    rows = np.flatnonzero(keep & np.isin(table["block"], editor_blocks))
    candidates = np.zeros((len(rows), 13, KNOTS, 9), np.float32)
    parent = np.zeros((len(rows), KNOTS, 9), np.float32)
    for k in range(FOLDS):
        left = np.load(output / "parents" / f"left_fold{k:02d}.npy", mmap_mode="r")
        mass_k = mass(left, np.flatnonzero(usable & (fold != k)))
        local = np.flatnonzero(fold[rows] == k)
        held = rows[local]
        pooled = np.asarray(left[held])
        parent[local] = blend(pooled, fnf[held], mass_k)
        extras = np.asarray(table["members"][held][:, [FNF_MEMBERS.index(m) for m in EXTRA_CANDIDATES]])
        ok = np.isfinite(extras).all(axis=(2, 3))
        extras = np.where(ok[:, :, None, None], extras, pooled[:, None])  # a failed extra: the pool
        candidates[local] = np.concatenate((np.asarray(table["core"][held]), extras), axis=1)
        say(stage="parents", fold=k, fnf_mass=mass_k, held=len(held))
    scale = np.asarray(table["magnitude"][rows])
    labels = np.asarray(table["labels"][rows])
    vocabulary = {label: g for g, label in enumerate(sorted(set(labels.tolist())))}
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / "index.npy", rows)
    np.save(cache / "candidates.npy", np.sort(candidates * scale[:, None, None, None], axis=-1))
    np.save(cache / "parent.npy", parent * scale[:, None, None])
    np.save(cache / "target.npy", np.asarray(table["target"][rows]) * scale[:, None])
    np.save(cache / "knot_weight.npy", np.asarray(table["knot_weight"][rows]))
    np.save(cache / "group.npy", np.asarray([vocabulary[label] for label in labels], np.int32))
    routing = {}
    for line in (output / "table" / "routing.jsonl").open():
        request = json.loads(line)
        routing[request.pop("index")] = request
    with (cache / "requests.jsonl").open("w") as stream:  # the requests, for routing prompts
        stream.writelines(json.dumps(routing[int(i)]) + "\n" for i in rows)
    (cache / "done").write_text(json.dumps(dict(rows=len(rows), fnf_mass=fnf_mass)) + "\n")
    return cache


def prepare(config: dict) -> None:
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    stage_forecasts(config, output)
    stage_table(config, output)
    in_waves(_fit, output, config)
    stage_parents(config, output)
    say(status="complete", router=str(output / "routers/full.ubj"), blend=str(output / "blend.json"),
        editor_cache=str(output / "editor_cache"))


# --------------------------------------------------------------------- editor
def sample_plan(groups: np.ndarray, *, updates: int, batch: int, seed: int) -> list[np.ndarray]:
    """Global batches: two random streams, each half uniform over windows and half uniform
    over frequency/horizon groups (so the plan does not depend on the number of GPUs)."""
    quarter = batch // 4
    members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    streams = [np.random.default_rng(seed + s) for s in range(2)]
    plan = []
    for _ in range(updates):
        parts = []
        for rng in streams:
            uniform = rng.choice(len(groups), size=quarter, replace=True)
            chosen = rng.integers(0, len(members), size=quarter)
            balanced = np.asarray([rng.choice(members[int(g)]) for g in chosen], dtype=np.int64)
            indices = np.concatenate((uniform, balanced)).astype(np.int64)
            rng.shuffle(indices)
            parts.append(indices)
        plan.append(np.concatenate(parts))
    return plan


def train_editor(config: dict, dist: Distributed) -> None:
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file

    from ..model.backbone import BASE_MODEL, BASE_REVISION, TARGET_MODULES, decoder, load_base_model
    from ..model.editor import AggregationEditor, editor_loss

    seed_everything(config.get("seed", 1))
    output = Path(config["output"])
    cache = {name: np.load(output / "editor_cache" / f"{name}.npy")
             for name in ("candidates", "parent", "target", "knot_weight", "group")}
    base = load_base_model(config.get("base_model", BASE_MODEL), config.get("base_revision", BASE_REVISION),
                           device=dist.device, gradient_checkpointing=True)
    lora = config.get("lora", dict(rank=4, alpha=8))
    model = get_peft_model(base, LoraConfig(r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=0.0,
                                            target_modules=list(TARGET_MODULES), bias="none",
                                            task_type="CAUSAL_LM"))
    model.train()  # enables gradient checkpointing (there is no dropout)
    editor = AggregationEditor(base.config.get_text_config().hidden_size).to(dist.device)
    parameters = [p for p in model.parameters() if p.requires_grad] + list(editor.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config.get("learning_rate", 3e-5),
                                  weight_decay=config.get("weight_decay", 0.0))
    batch, micro = config.get("global_batch_size", 32), config.get("micro_batch_size", 8)
    plan = sample_plan(cache["group"], updates=config.get("updates", 128), batch=batch, seed=config.get("seed", 1))
    tensor = lambda name, rows: torch.as_tensor(cache[name][rows], device=dist.device)
    for update, indices in enumerate(plan):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        local = indices[dist.rank::dist.world_size]
        for begin in range(0, len(local), micro):
            rows = local[begin:begin + micro]
            _, correction = editor(decoder(model), tensor("candidates", rows), tensor("parent", rows))
            loss = editor_loss(correction, target=tensor("target", rows), reference=tensor("parent", rows),
                               candidates=tensor("candidates", rows), knot_weight=tensor("knot_weight", rows),
                               beta=config.get("beta", 0.05), edit_weight=config.get("edit_weight", 0.2))
            (loss.total / len(indices)).backward()
            total += float(loss.total.detach())
        dist.sync_gradients(parameters)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        loss = dist.all_reduce(torch.tensor(total, device=dist.device)).item() / len(indices)
        log(dist, update=update + 1, of=len(plan), loss=loss, grad_norm=float(grad_norm))
    if dist.main:
        expert = output / "expert"
        model.save_pretrained(expert / "adapter")
        save_file({k: v.detach().cpu().contiguous() for k, v in editor.state_dict().items()},
                  str(expert / "editor.safetensors"))
        print(f"saved aggregation expert to {expert}", flush=True)
    dist.barrier()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("prepare", "editor"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config, dict(output=args.output))
    if args.stage == "prepare":
        prepare(config)
    else:
        train_editor(config, setup_distributed())


if __name__ == "__main__":
    main()
