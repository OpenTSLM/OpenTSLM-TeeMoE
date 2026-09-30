"""GIFT-Eval: mean MASE rank among the leaderboard's models over the 97 test cells.

    python -m teemoe.eval.gift --gift-data data/gift-eval --output results/gift --gpus 0 1 2 3 4 5 6 7

Stages, each reused when complete:

  panel  every test window (371,330): history (at most 8,192 observations, gaps
         interpolated), horizon, target and the history's seasonal MASE scale
  run    TeeMoE on the windows, one worker per GPU (each worker also runs the
         forecasting models and vLLM on its GPU). As in the paper, the Toto-FnF
         forecast is the one released with the FnF bundle (member forecasts,
         features and dataset metadata; about 17 GB, downloaded once)
  score  MASE and CRPS per cell, ranked against the leaderboard's result files
         in the GIFT-Eval checkout (pinned revision)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from ..data.windows import PROPERTY_ALIASES, gift_dataset, gift_eval, gift_metadata, timestamp
from ..ensemble.forecasters import child_gpus
from ..runtime import wait_workers, worker_pool
from .common import add_model_arguments, save_report

LEVELS = np.arange(1, 10) / 10
MAX_HISTORY = 8192
CHUNK = 16384  # windows per forecast_batch call (each call loads the forecasting models once)


def frequency(name: str, raw: str) -> str:
    """The cell's frequency as the leaderboard names it (e.g. 'W', 'A', '15T')."""
    if "/" in name:
        return name.split("/", 1)[1]
    base = raw.split("-", 1)[0]
    return {"Y": "A", "YE": "A", "QE": "Q", "ME": "M", "h": "H", "min": "T", "s": "S"}.get(base, base)


def history(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).copy()
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        values = np.zeros(max(len(values), 8), np.float32)
    elif len(finite) < len(values):
        missing = np.flatnonzero(~np.isfinite(values))
        values[missing] = np.interp(missing, finite, values[finite])
    if len(values) < 4:
        values = np.pad(values, (4 - len(values), 0), mode="edge")
    return values[-MAX_HISTORY:]


# ---------------------------------------------------------------------- panel
def build_panel(root: Path, data: Path, output: Path) -> Path:
    panel = output / "panel"
    if (panel / "cells.json").exists():
        return panel
    from gluonts.ev.ts_stats import seasonal_error
    from gluonts.time_feature import get_seasonality

    Dataset, cells, properties = gift_eval(root, data)
    histories, futures, meta, scales, names = [], [], [], [], []
    for name, term in cells:
        dataset, variates = gift_dataset(Dataset, name, term)
        freq = frequency(name, str(dataset.freq))
        base = name.split("/")[0].lower()
        cell = f"{PROPERTY_ALIASES.get(base, base)}/{freq}/{term}"
        seasonality = get_seasonality(dataset.freq)
        domain = gift_metadata(properties, name, variates)["domain"]
        for observed, label in zip(dataset.test_data.input, dataset.test_data.label, strict=True):
            h = history(observed["target"])
            scale = seasonal_error(np.ma.masked_invalid(np.asarray(observed["target"])), seasonality=seasonality,
                                   time_axis=-1).reshape(-1)[0]
            valid = not np.ma.is_masked(scale) and np.isfinite(scale) and float(scale) > 0
            histories.append(h)
            futures.append(np.asarray(label["target"], np.float32))
            scales.append(float(scale) if valid else np.nan)
            meta.append(dict(frequency=freq, start=timestamp(observed["start"] + (len(observed["target"]) - len(h))),
                             term=term, domain=domain, dataset=cell.split("/")[0], cell=len(names)))
        names.append(cell)
        print(json.dumps(dict(stage="panel", cell=cell, windows=len(meta))), flush=True)
    panel.mkdir(parents=True, exist_ok=True)
    for key, arrays in (("history", histories), ("future", futures)):
        np.save(panel / f"{key}.npy", np.concatenate(arrays))
        np.save(panel / f"{key}_offsets.npy", np.concatenate(([0], np.cumsum([len(a) for a in arrays]))))
    np.save(panel / "scale.npy", np.asarray(scales, np.float64))
    with (panel / "requests.jsonl").open("w") as stream:
        stream.writelines(json.dumps(m) + "\n" for m in meta)
    (panel / "cells.json").write_text(json.dumps(names) + "\n")
    return panel


class Panel:
    def __init__(self, path: Path) -> None:
        self.meta = [json.loads(line) for line in (path / "requests.jsonl").open()]
        load = lambda name: np.load(path / f"{name}.npy", mmap_mode="r")
        self.history, self.history_offsets = load("history"), load("history_offsets")
        self.future, self.future_offsets = load("future"), load("future_offsets")
        self.scale = np.load(path / "scale.npy")
        self.cells = json.loads((path / "cells.json").read_text())

    def __len__(self) -> int:
        return len(self.meta)

    def request(self, i: int) -> dict:
        h = self.history[self.history_offsets[i]:self.history_offsets[i + 1]]
        horizon = int(self.future_offsets[i + 1] - self.future_offsets[i])
        return dict(self.meta[i], history=np.asarray(h, np.float64).tolist(), horizon=horizon)

    def target(self, i: int) -> np.ndarray:
        return np.asarray(self.future[self.future_offsets[i]:self.future_offsets[i + 1]], np.float64)


# ------------------------------------------------------------------------ run
def prepare_run(args) -> None:
    """Resume only when the checkpoint files and evaluation settings still match."""
    from ..checkpoint import resolve
    from ..model.mixture import EXPERTS

    root = resolve(args.checkpoint).resolve()
    files = [root / name for name in ("teemoe.json", "editor.safetensors", "controller.safetensors", "router.ubj")]
    files += [root / "adapters" / expert / name for expert in EXPERTS
              for name in ("adapter_config.json", "adapter_model.safetensors")]
    signatures = {}
    for path in files:
        info = path.stat()
        signatures[str(path.relative_to(root))] = [info.st_size, info.st_mtime_ns, info.st_ctime_ns]
    settings = dict(checkpoint=str(root), checkpoint_files=signatures, workers=len(args.gpus), chunk_size=CHUNK,
                    backend=args.backend, model_class=os.environ.get("TEEMOE_MODEL_CLASS"),
                    model_options=json.loads(os.environ.get("TEEMOE_MODEL_OPTIONS", "{}")),
                    vllm_python=args.vllm_python or os.environ.get("TEEMOE_VLLM_PYTHON"),
                    environments=json.loads(Path(args.environments).read_text()) if args.environments else None)
    for name in ("gift_root", "gift_data", "fnf_root", "granite"):
        value = getattr(args, name)
        settings[name] = str(Path(value).resolve()) if value else None
    output = Path(args.output)
    saved = output / "settings.json"
    if saved.exists():
        if json.loads(saved.read_text()) != settings:
            raise ValueError("GIFT output belongs to a different checkpoint or evaluation configuration. "
                             "Use a new --output directory; existing results have not been changed.")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("GIFT output has files without compatible run settings. Use a new --output directory.")
        output.mkdir(parents=True, exist_ok=True)
        with saved.open("x") as stream:
            json.dump(settings, stream, indent=2)
            stream.write("\n")
    args.checkpoint = str(root)


def work(args, worker: int, workers: int) -> None:
    from .common import load_model

    from ..ensemble.fnf import TotoFnF

    panel, output = Panel(Path(args.output) / "panel"), Path(args.output) / "run"
    output.mkdir(parents=True, exist_ok=True)
    rows = np.arange(worker, len(panel), workers)
    cells = np.array([m["cell"] for m in panel.meta])
    first, count = np.searchsorted(cells, np.arange(len(panel.cells))), np.bincount(cells)
    fnf, released = TotoFnF(args.fnf_root), {}
    model = None
    for number, begin in enumerate(range(0, len(rows), CHUNK)):
        path = output / f"worker{worker:02d}-{number:04d}.npz"
        if path.exists():
            continue
        model = model or load_model(args)
        chunk = rows[begin:begin + CHUNK]
        needed = set(cells[chunk].tolist())
        released = {c: released[c] if c in released else fnf.released(panel.cells[c]) for c in needed}
        if any(len(released[c]) != count[c] for c in needed):
            raise ValueError("the released FnF forecasts do not cover the panel's windows")
        requests = [dict(panel.request(int(i)), fnf_forecast=released[cells[i]][i - first[cells[i]]]) for i in chunk]
        forecasts = model.forecast_batch(requests, seed=int(chunk[0]))
        np.savez(path.with_suffix(".tmp.npz"), index=chunk, weights=np.stack([f.weights for f in forecasts]),
                 numerical=np.asarray([f.output == "numerical" for f in forecasts]),
                 quantiles=np.concatenate([f.quantiles for f in forecasts]).astype(np.float32))
        path.with_suffix(".tmp.npz").rename(path)
        print(json.dumps(dict(stage="run", worker=worker, rows=begin + len(chunk), of=len(rows))), flush=True)


def launch(args) -> None:
    gpus = [str(g) for g in args.gpus]
    with worker_pool() as processes:
        for worker, gpu in enumerate(gpus):
            command = [sys.executable, "-m", "teemoe.eval.gift", "--worker", str(worker), "--workers", str(len(gpus)),
                       "--device", "cuda:0", "--vllm-devices", "0", "--forecast-devices", "cuda:0"]
            for key in ("checkpoint", "backend", "vllm_python", "environments", "granite", "fnf_root", "output",
                        "gift_root", "gift_data"):
                if getattr(args, key) is not None:
                    command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
            processes.append(subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=child_gpus(gpu)),
                                              start_new_session=True))
        wait_workers(processes)


# ---------------------------------------------------------------------- score
def leaderboard(root: Path) -> dict[str, dict[str, float]]:
    """Each leaderboard model's MASE per cell, from the checkout's result files."""
    results = {}
    for path in sorted((root / "results").glob("*/all[_-]results.csv")):
        with path.open(newline="") as stream:
            results[path.parent.name] = {row["dataset"]: float(row["eval_metrics/MASE[0.5]"])
                                         for row in csv.DictReader(stream)}
    return results


def score(args) -> dict:
    panel = Panel(Path(args.output) / "panel")
    quantiles, numerical = [None] * len(panel), np.zeros(len(panel), bool)
    seen = np.zeros(len(panel), bool)
    for path in sorted((Path(args.output) / "run").glob("worker*-*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        with np.load(path) as part:
            indices = part["index"]
            values, flags = part["quantiles"], part["numerical"]
            if (indices.ndim != 1 or indices.dtype.kind not in "iu" or np.any(indices < 0)
                    or np.any(indices >= len(panel)) or len(np.unique(indices)) != len(indices)
                    or seen[indices].any()):
                raise ValueError(f"Duplicate or invalid GIFT window indices in {path.name}")
            offsets = np.cumsum([0] + [len(panel.target(int(i))) for i in indices])
            if values.shape != (offsets[-1], len(LEVELS)) or flags.shape != indices.shape:
                raise ValueError(f"Incomplete GIFT forecasts in {path.name}")
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite GIFT forecasts in {path.name}")
            for j, i in enumerate(indices):
                quantiles[int(i)] = values[offsets[j]:offsets[j + 1]]
            numerical[indices] = flags
            seen[indices] = True
    missing = sum(q is None for q in quantiles)
    if missing:
        raise RuntimeError(f"{missing} windows have no forecast; rerun the evaluation to complete them")
    totals = np.zeros((len(panel.cells), 3 + len(LEVELS)))  # MASE sum, MASE count, |target| sum, pinball sums
    for i, q in enumerate(quantiles):
        y = panel.target(i)
        mask = np.isfinite(y)
        y = np.where(mask, y, 0.0)
        row = totals[panel.meta[i]["cell"]]
        if np.isfinite(panel.scale[i]):
            row[0] += (np.abs(q[:, 4] - y) * mask).sum() / panel.scale[i]
            row[1] += mask.sum()
        row[2] += (np.abs(y) * mask).sum()
        error = y[:, None] - q
        row[3:] += (2 * np.maximum(LEVELS * error, (LEVELS - 1) * error) * mask[:, None]).sum(0)
    if not np.isfinite(totals).all() or np.any(totals[:, 1] <= 0):
        raise ValueError("GIFT metrics require finite totals and valid MASE observations in every cell")
    mase = dict(zip(panel.cells, totals[:, 0] / totals[:, 1], strict=True))
    crps = (totals[:, 3:] / np.maximum(totals[:, 2:3], 1e-8)).mean(1)
    if not np.isfinite(list(mase.values())).all() or not np.isfinite(crps).all():
        raise ValueError("Nonfinite GIFT metrics cannot be ranked")
    peers = {name: cells for name, cells in leaderboard(Path(args.gift_root)).items() if set(mase) <= set(cells)}
    if not peers:
        raise ValueError("No complete leaderboard entries are available for GIFT ranking")
    for name, cells in peers.items():
        if any(not np.isfinite(cells[cell]) for cell in mase):
            raise ValueError(f"Nonfinite leaderboard MASE for {name}")
    rank = lambda cell, value: (1 + sum(p[cell] < value for p in peers.values())
                                + 0.5 * sum(p[cell] == value for p in peers.values()))  # ties count half
    ranks = {cell: rank(cell, value) for cell, value in mase.items()}
    return dict(benchmark="GIFT-Eval", windows=len(panel), cells=len(mase), leaderboard_models=len(peers),
                mean_mase_rank=float(np.mean(list(ranks.values()))),
                geometric_mean_mase=float(np.exp(np.mean(np.log(np.maximum(list(mase.values()), 1e-12))))),
                geometric_mean_crps=float(np.exp(np.mean(np.log(np.maximum(crps, 1e-12))))),
                numerical_fraction=float(numerical.mean()),
                cells_detail={cell: dict(mase=mase[cell], crps=float(c), rank=ranks[cell])
                              for cell, c in zip(panel.cells, crps, strict=True)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser)
    parser.add_argument("--gift-root", default="third_party/gift-eval", help="GIFT-Eval checkout (pinned revision)")
    parser.add_argument("--gift-data", default="data/gift-eval", help="Salesforce/GiftEval dataset directory")
    parser.add_argument("--output", default="results/gift")
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--workers", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        return work(args, args.worker, args.workers)
    from ..ensemble.fnf import ensure_fnf

    prepare_run(args)
    build_panel(Path(args.gift_root), Path(args.gift_data), Path(args.output))
    args.fnf_root = str(ensure_fnf(args.fnf_root, gift=True))
    launch(args)
    report = score(args)
    save_report(Path(args.output) / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "cells_detail"}, indent=2))


if __name__ == "__main__":
    main()
