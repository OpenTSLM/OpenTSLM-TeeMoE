"""XGBoost weighting of the eight core forecasters and CDF pooling (paper Sec. 3.2, App. A.5)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CORE = 8
QUANTILES = 9
GLOBAL = 19  # global features in the 195-value router input
LOCAL = 22   # per-forecaster features


@dataclass(frozen=True)
class RouterRecipe:
    rounds: int = 1200
    max_depth: int = 10
    eta: float = 0.04
    min_child_weight: float = 12.0
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 2.0
    reg_alpha: float = 0.05
    max_bin: int = 256
    temperature: float = 1.35
    score_std_floor: float = 0.15
    seed: int = 0
    nthread: int = 16


def frequency_seasonality(frequency: str) -> int:
    match = re.match(r"^(\d+)?([A-Za-z]+)", str(frequency).strip())
    if match is None:
        raise ValueError(f"unsupported frequency {frequency!r}")
    aliases = {"second": "S", "secondly": "S", "minute": "T", "minutely": "T", "min": "T",
               "hour": "H", "hourly": "H", "day": "D", "daily": "D", "week": "W", "weekly": "W",
               "month": "M", "monthly": "M", "quarter": "Q", "quarterly": "Q", "year": "Y",
               "yearly": "Y", "annual": "Y"}
    base = aliases.get(match[2].lower(), match[2].upper())
    seasons = {"S": 3600, "T": 1440, "H": 24, "B": 5, "M": 12, "MS": 12, "ME": 12,
               "Q": 4, "QS": 4, "QE": 4}
    period, remainder = divmod(seasons.get(base, 1), int(match[1] or 1))
    return max(period, 1) if remainder == 0 else 1


def _frequency_family(frequency: str) -> str:
    value = str(frequency).upper()
    if "MIN" in value or value.endswith("T"):
        return "minute"
    for prefixes, name in (("H", "hour"), ("DB", "day"), ("W", "week"), ("M", "month"),
                           ("Q", "quarter"), ("AY", "year")):
        if value.startswith(tuple(prefixes)):
            return name
    return "other"


def _slope(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    x = np.arange(values.size, dtype=np.float64)
    x -= x.mean()
    return float(np.dot(x, values - values.mean()) / max(float(np.dot(x, x)), 1e-12))


def _rank01(values: np.ndarray) -> np.ndarray:
    less = (values[:, None] > values[None, :]).sum(axis=1)
    equal = (values[:, None] == values[None, :]).sum(axis=1)
    return (less + 0.5 * (equal - 1)).astype(np.float64) / (len(values) - 1)


def router_features(history, quantiles, *, frequency: str, term: str,
                    history_context: int = 512) -> np.ndarray:
    """195 values from the recent history and the core forecasts [8, horizon, 9]."""
    values = np.asarray(history, dtype=np.float64)[-history_context:]
    values = values[np.isfinite(values)]
    q = np.asarray(quantiles, dtype=np.float64)
    if not values.size or q.ndim != 3 or q.shape[0] != CORE or q.shape[2] != QUANTILES \
            or not np.isfinite(q).all():
        raise ValueError("expected a finite history and eight finite nine-quantile forecasts")
    horizon = q.shape[1]
    magnitude = max(float(np.max(np.abs(values))), 1.0)
    scaled = values / magnitude
    location = float(np.median(scaled))
    centered = scaled - location
    mad = float(np.median(np.abs(centered))) * 1.4826
    roughness = float(np.mean(np.abs(np.diff(scaled)))) if scaled.size > 1 else 0.0
    scale = max(mad, roughness, float(centered.std()), abs(location) * 1e-4, 1e-6)
    normalized = np.clip(centered / scale, -50.0, 50.0)
    q = np.clip((q / magnitude - location) / scale, -100.0, 100.0)
    medians, spreads = q[..., 4], q[..., -1] - q[..., 0]
    asymmetry = q[..., -1] + q[..., 0] - 2.0 * medians
    seasonality = frequency_seasonality(frequency)
    family = _frequency_family(frequency)
    features = [math.log1p(values.size), math.log1p(horizon),
                math.log((values.size + 1.0) / (horizon + 1.0)), math.log1p(magnitude),
                location, scale, math.log1p(seasonality), float(horizon / seasonality)]
    features += [float(family == name) for name in
                 ("minute", "hour", "day", "week", "month", "quarter", "year", "other")]
    features += [float(str(term).lower() == name) for name in ("short", "medium", "long")]
    trend = np.clip(normalized[-1] + _slope(normalized[-min(64, normalized.size):])
                    * np.arange(1, horizon + 1, dtype=np.float64), -100.0, 100.0)
    seasonal = (np.asarray([normalized[-seasonality + i % seasonality] for i in range(horizon)])
                if normalized.size >= seasonality else np.full(horizon, normalized[-1]))
    consensus = np.median(medians, axis=0)
    rough = np.mean(np.abs(np.diff(medians, axis=1)), axis=1) if horizon > 1 else np.zeros(CORE)
    ranks = (_rank01(medians.mean(axis=1)), _rank01(spreads.mean(axis=1)), _rank01(rough),
             _rank01(np.mean(np.abs(medians - consensus[None, :]), axis=1)))
    for e in range(CORE):
        median, spread = medians[e], spreads[e]
        curvature = np.diff(median, n=2)
        features += [float(median.mean()), float(median.std()), float(median[0]), float(median[-1]),
                     _slope(median), float(rough[e]),
                     float(np.mean(np.abs(curvature))) if curvature.size else 0.0,
                     float(spread.mean()), float(spread.std()), float(spread.min()), float(spread.max()),
                     float(spread[0]), float(spread[-1]), float(asymmetry[e].mean()),
                     float(asymmetry[e].std()), float(median[0] - normalized[-1]),
                     float(np.mean(np.abs(median - trend))), float(np.mean(np.abs(median - seasonal))),
                     *(float(rank[e]) for rank in ranks)]
    return np.asarray(features, dtype=np.float32)


def expert_features(selected: np.ndarray) -> np.ndarray:
    """Expand [rows, 195] router inputs to one 1,433-value row per (window, forecaster)."""
    x = np.asarray(selected, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != GLOBAL + CORE * LOCAL or not np.isfinite(x).all():
        raise ValueError("router inputs must be finite [rows, 195]")
    rows = len(x)
    global_x = x[:, :GLOBAL]
    local = x[:, GLOBAL:].reshape(rows, CORE, LOCAL)
    mean, std = local.mean(1), local.std(1)
    lo, hi = local.min(1), local.max(1)
    ordered = np.sort(local, axis=1)
    order = np.argsort(local, axis=1, kind="stable")
    rank = np.empty_like(order, dtype=np.float32)
    np.put_along_axis(rank, order, np.broadcast_to(
        np.arange(CORE, dtype=np.float32)[None, :, None], order.shape), axis=1)
    flat = local.reshape(rows, -1)
    shared = [mean, hi - lo, std, lo, hi, 0.5 * (ordered[:, 3] + ordered[:, 4])]
    identity = np.eye(CORE, dtype=np.float32)
    pieces = []
    for e in range(CORE):
        focal = local[:, e]
        difference = focal[:, None, :] - local
        pieces.append(np.concatenate((
            global_x, focal, flat, np.tile(focal, (1, CORE)),
            difference.reshape(rows, -1), np.abs(difference).reshape(rows, -1), *shared,
            (focal - mean) / np.maximum(std, 0.05), rank[:, e] / 7.0,
            (rank / 7.0).reshape(rows, -1),
            np.broadcast_to(identity[e], (rows, CORE)),
            (global_x[:, :, None] * identity[e][None, None, :]).reshape(rows, -1),
            ordered.reshape(rows, -1)), axis=1))
    return np.stack(pieces, axis=1).reshape(rows * CORE, -1).astype(np.float32)


def rank_labels(ranks: np.ndarray) -> np.ndarray:
    """Regression target log(1 + rank), rank 0 = lowest empirical CRPS in the window."""
    return np.log1p(np.asarray(ranks, dtype=np.float32)).reshape(-1)


def allocations(scores: np.ndarray, *, temperature: float = 1.35, floor: float = 0.15) -> np.ndarray:
    """w = softmax(-z / T) with z the row-standardized predicted log-ranks."""
    score = np.asarray(scores, dtype=np.float64).reshape(-1, CORE)
    score -= score.mean(axis=1, keepdims=True)
    score /= np.maximum(score.std(axis=1, keepdims=True), floor)
    logits = -score / temperature
    logits -= logits.max(axis=1, keepdims=True)
    weight = np.exp(logits)
    return (weight / weight.sum(axis=1, keepdims=True)).astype(np.float32)


def pool_quantiles(quantiles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Quantiles of the weighted mixture of the candidates' discrete CDFs ([rows, 8, h, 9] -> [rows, h, 9])."""
    q = np.asarray(quantiles, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(q).all(axis=(2, 3)) & (np.abs(q) < 1e30).all(axis=(2, 3))
    w = np.where(valid, w, 0.0)
    w /= w.sum(1, keepdims=True)
    q = np.where(valid[:, :, None, None], q, np.float32(0))
    atoms = q.transpose(0, 2, 1, 3).reshape(len(q), q.shape[2], -1)
    order = np.argsort(atoms, axis=2, kind="stable")
    values = np.take_along_axis(atoms, order, axis=2)
    mass = np.take_along_axis(np.broadcast_to(w[:, None], (len(q), q.shape[2], CORE)),
                              order // QUANTILES, axis=2) / QUANTILES
    cumulative = np.cumsum(mass, axis=2, dtype=np.float64)
    levels = np.arange(1, 10) / 10
    result = [np.take_along_axis(values, np.argmax(cumulative >= p, axis=2)[..., None], axis=2)[..., 0]
              for p in levels]
    return np.stack(result, axis=-1).astype(np.float32)


class Router:
    """A fitted core-8 router: selected features -> allocations -> pooled quantiles."""

    def __init__(self, path: str | Path, recipe: RouterRecipe = RouterRecipe(), device: str = "cpu"):
        import xgboost as xgb

        self.recipe = recipe
        self.booster = xgb.Booster()
        self.booster.load_model(str(path))
        self.booster.set_param({"device": device, "nthread": recipe.nthread})

    def allocations(self, selected: np.ndarray, chunk: int = 4096) -> np.ndarray:
        parts = [self.booster.inplace_predict(expert_features(selected[i:i + chunk]))
                 for i in range(0, len(selected), chunk)]
        return allocations(np.concatenate(parts), temperature=self.recipe.temperature,
                           floor=self.recipe.score_std_floor)

    def pool(self, selected: np.ndarray, core_quantiles: np.ndarray) -> np.ndarray:
        return pool_quantiles(core_quantiles, self.allocations(selected))


def fit_router(selected: np.ndarray, ranks: np.ndarray, output: str | Path,
               recipe: RouterRecipe = RouterRecipe(), device: str = "cuda", chunk: int = 16384) -> Path:
    """Fit the rank regressor on [rows, 195] inputs and [rows, 8] CRPS ranks (expanded in chunks)."""
    import xgboost as xgb

    class Chunks(xgb.DataIter):
        def __init__(self) -> None:
            self.position = 0
            super().__init__()

        def next(self, feed) -> bool:
            if self.position >= len(selected):
                return False
            end = min(self.position + chunk, len(selected))
            feed(data=expert_features(np.asarray(selected[self.position:end])),
                 label=rank_labels(np.asarray(ranks[self.position:end])))
            self.position = end
            return True

        def reset(self) -> None:
            self.position = 0

    matrix = xgb.QuantileDMatrix(Chunks(), max_bin=recipe.max_bin, nthread=recipe.nthread)
    params = dict(objective="reg:squarederror", tree_method="hist", device=device,
                  max_depth=recipe.max_depth, eta=recipe.eta, min_child_weight=recipe.min_child_weight,
                  subsample=recipe.subsample, colsample_bytree=recipe.colsample_bytree,
                  reg_lambda=recipe.reg_lambda, alpha=recipe.reg_alpha, max_bin=recipe.max_bin,
                  seed=recipe.seed, nthread=recipe.nthread)
    booster = xgb.Booster(params, [matrix])
    for iteration in range(recipe.rounds):
        booster.update(matrix, iteration)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(output))
    return output


def crps_ranks(quantiles: np.ndarray, targets: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Rank the 8 core forecasts [rows, 8, knots, 9] per window by scaled empirical CRPS."""
    q = np.asarray(quantiles, np.float32)
    y = np.asarray(targets, np.float32)
    first = np.mean(np.abs(q - y[:, None, :, None]), axis=(2, 3))
    coefficients = np.arange(-8, 9, 2, dtype=np.float32) / 81.0
    second = np.mean(np.sum(q * coefficients, axis=3), axis=2)
    risk = np.maximum(first - second, 0.0) / np.maximum(np.asarray(scales, np.float32)[:, None], 1e-6)
    order = np.argsort(risk, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.uint8)
    ranks[np.arange(len(ranks))[:, None], order] = np.arange(CORE, dtype=np.uint8)
    return ranks
