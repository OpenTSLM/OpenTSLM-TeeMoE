"""Context is Key: weighted RCRPS over its 71 tasks at instance seeds 1-5 (355 forecasts).

    python -m teemoe.eval.cik --cik-root third_party/context-is-key --output results/cik.json

Every forecast is 25 trajectories: sampled from the native expert, or drawn from
the numerical forecast's quantiles when the controller chooses it. Scores use
the benchmark's own region-of-interest and constraint metric with a numerically
stable empirical CRPS, each capped at 5, averaged with the benchmark's weights.
The benchmark downloads its source data on first use (``CIK_DATA_STORE``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import types
from pathlib import Path

import numpy as np

from ..prompts import format_value, render_user
from .common import add_model_arguments, load_model, save_report

CIK = dict(repository="https://github.com/ServiceNow/context-is-key-forecasting.git",
           revision="73f46016f8c5643bf6799ca31875eab3e8d0d075")
SEEDS = (1, 2, 3, 4, 5)
SAMPLES = 25
CAP = 5.0


def import_benchmark(root: Path):
    os.environ.setdefault("CIK_DATA_STORE", str(root.resolve() / "data"))
    sys.path.insert(0, str(root.resolve()))
    stub = types.ModuleType("cik_benchmark.baselines.lag_llama")  # an optional baseline we do not need
    stub.get_lag_llama_predictions = stub.prepare_dataset = stub.format_llama_predictions = None
    sys.modules.setdefault("cik_benchmark.baselines.lag_llama", stub)
    import cik_benchmark

    return cik_benchmark


def stable_crps(target: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """The benchmark's unbiased empirical CRPS, summed over sample pairs as nonnegative slacks
    |a - y| + |b - y| - |a - b| (the moment form cancels badly for very large values)."""
    y, s = np.asarray(target, np.float64), np.asarray(samples, np.float64)
    total = np.zeros(y.shape)
    for i in range(len(s) - 1):
        lower, upper = np.minimum(s[i], s[i + 1:]), np.maximum(s[i], s[i + 1:])
        total += np.where(upper <= y, 2 * (y - upper), np.where(lower >= y, 2 * (lower - y), 0.0)).sum(0)
    return total / (len(s) * (len(s) - 1))


def task_request(task, name: str, seed: int) -> dict:
    """The observed history, supplied context and requested timestamps (never the future values)."""
    past = task.past_time.iloc[:, -1]
    stamps = lambda index: list(index.strftime("%Y-%m-%d %H:%M:%S"))
    recent, future = past.iloc[-168:], stamps(task.future_time.index)
    history = "\n".join(f"({t}, {format_value(v)})" for t, v in zip(stamps(recent.index), recent.values, strict=True))
    context = "".join(f"{field.capitalize()}: {value}\n" for field in ("background", "constraints", "scenario")
                      if (value := getattr(task, field, None))).strip()
    message = render_user(dict(mold="context_aided_forecast", context=context, evidence=history,
                               history_evidence=history, prediction_points="\n".join(future)))
    frequency = (past.index.freqstr or past.index.inferred_freq or task.future_time.index.freqstr
                 or task.future_time.index.inferred_freq)
    digest = hashlib.blake2b(f"{name}|{seed}".encode(), digest_size=8).digest()
    return dict(history=past.astype(float).tolist(), horizon=len(future), frequency=frequency,
                start=str(past.index[0]), past_timestamps=stamps(past.index), future_timestamps=future,
                message=message, seed=int.from_bytes(digest, "little") % (2**31))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser)
    parser.add_argument("--cik-root", default="third_party/context-is-key", help="benchmark checkout")
    parser.add_argument("--output", default="results/cik.json")
    args = parser.parse_args()
    cik = import_benchmark(Path(args.cik_root))
    from cik_benchmark.metrics import roi_metric

    tasks = sorted(cik.ALL_TASKS, key=lambda t: t.__name__)
    worlds = [(t.__name__, seed, t(seed=seed)) for t in tasks for seed in SEEDS]
    requests = [task_request(task, name, seed) for name, seed, task in worlds]
    forecasts = load_model(args).forecast_batch(requests, samples=SAMPLES)
    roi_metric.crps = stable_crps
    records = []
    for (name, seed, task), request, forecast in zip(worlds, requests, forecasts, strict=True):
        paths = forecast.trajectories(SAMPLES, request["seed"])
        evaluated = task.evaluate(np.asarray(paths, np.float64)[:, :, None])
        metric = float(evaluated["metric"] if isinstance(evaluated, dict) else evaluated)
        if not math.isfinite(metric):
            raise ValueError(f"nonfinite score for {name} seed {seed}")
        records.append(dict(task=name, seed=seed, weight=float(cik.TASK_NAME_TO_WEIGHT[name]), metric=metric,
                            output=forecast.output, weights=forecast.weights.tolist()))
    total = sum(r["weight"] for r in records)
    score = sum(r["weight"] * min(r["metric"], CAP) for r in records) / total
    save_report(args.output, dict(benchmark="Context is Key", instances=len(records), weighted_rcrps=score,
                                  numerical_outputs=sum(r["output"] == "numerical" for r in records),
                                  records=records))
    print(json.dumps(dict(instances=len(records), weighted_rcrps=score)))


if __name__ == "__main__":
    main()
