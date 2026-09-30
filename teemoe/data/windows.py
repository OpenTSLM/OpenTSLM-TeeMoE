"""Forecasting windows from public time-series collections, shared by the data builders.

A *window* is a JSON-serializable dict with at least ``history``, ``future``,
``freq``, ``term``, ``dataset`` and a physical-series identity (``corpus``,
``dataset``, ``source_series``) used to keep related windows in one
cross-fitting fold.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

GIFT_EVAL = dict(repository="https://github.com/SalesforceAIResearch/gift-eval.git",
                 revision="4d5ab3fa0fe7451bbf59bb1ff6dd76e6e414d64a")
# GIFT-Eval datasets with medium and long terms (all others are short only).
LONG_TERMS = set(
    "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T "
    "LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T "
    "jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service "
    "bizitobs_l2c/5T bizitobs_l2c/H".split())
PROPERTY_ALIASES = {"saugeenday": "saugeen", "temperature_rain_with_missing": "temperature_rain",
                    "kdd_cup_2018_with_missing": "kdd_cup_2018", "car_parts_with_missing": "car_parts"}
# Observed histories that also occur in the GIFT-Eval test split; never used for training.
EXCLUDED_HISTORIES = {bytes.fromhex("a3c8f0385edde5c612521a693d19f5d4")}


# ------------------------------------------------------------------ utilities
def stable_seed(seed: int, *parts) -> int:
    payload = "\0".join([str(seed), *map(str, parts)]).encode()
    return seed ^ int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def repair(values) -> np.ndarray | None:
    """Linearly interpolate missing values (None when nothing is observed)."""
    values = np.asarray(values, dtype=np.float32).copy()
    finite = np.flatnonzero(np.isfinite(values))
    if not finite.size:
        return None
    missing = np.flatnonzero(~np.isfinite(values))
    if missing.size:
        values[missing] = np.interp(missing, finite, values[finite])
    return values


def history_digest(history) -> bytes:
    values = np.asarray(history, dtype=np.float32)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.asarray([len(values)], dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.digest()


def content_digest(row: dict) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    for key in ("history", "future"):
        values = np.asarray(row[key], dtype=np.float32)
        digest.update(np.asarray([len(values)], dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.digest()


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary, count = path.with_suffix(path.suffix + ".tmp"), 0
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def read_jsonl(path: str | Path, shard: int = 0, shards: int = 1) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index % shards == shard and line.strip():
                yield json.loads(line)


def log(**values) -> None:
    print(json.dumps(values), flush=True)


# -------------------------------------------------------- window sampling rule
def frequency_regime(frequency: str) -> str:
    value = str(frequency)
    legacy = re.fullmatch(r"(\d*)([STHMQAY])", value, flags=re.IGNORECASE)
    if legacy:
        multiplier, unit = legacy.groups()
        value = multiplier + {"S": "s", "T": "min", "H": "h", "M": "ME", "Q": "QE", "A": "YE",
                              "Y": "YE"}[unit.upper()]
    from pandas.tseries.frequencies import to_offset

    name = str(to_offset(value).name).lower()
    for prefixes, single, regime in ((("ye", "y-", "a-"), "a", "year"), (("qe", "q-"), "q", "quarter"),
                                     (("me",), "m", "month"), (("w",), "w", "week")):
        if name.startswith(prefixes) or name == single:
            return regime
    units = dict(d="day", h="hour", min="minute", t="minute", s="second", sec="second")
    if name not in units:
        raise ValueError(f"unsupported frequency {frequency!r}")
    return units[name]


def term_horizons(frequency: str, max_horizon: int = 900) -> tuple[tuple[str, int], ...]:
    """GIFT-Eval style short / medium / long horizons (1x, 10x, 15x a base horizon)."""
    base = dict(year=6, quarter=8, month=12, week=8, day=30, hour=48, minute=48,
                second=60)[frequency_regime(frequency)]
    return tuple((term, base * m) for term, m in (("short", 1), ("medium", 10), ("long", 15))
                 if base * m <= max_horizon)


def admissible(history: np.ndarray, future: np.ndarray) -> bool:
    """Finite, not all zero, and no target far outside the history's range of variation."""
    if not len(history) or not len(future) or not np.isfinite(history).all() \
            or not np.isfinite(future).all() or not np.any(history != 0):
        return False
    mean = float(np.mean(np.abs(history), dtype=np.float64))
    std = float(np.std(history, dtype=np.float64))
    step = float(np.mean(np.abs(np.diff(history)), dtype=np.float64)) if len(history) > 1 else 0.0
    return float(np.max(np.abs(future))) / max(mean, std, 4.0 * step, 1.0) <= 1e6


def sample_windows(values, *, frequency: str, dataset: str, series_id: str, group: str, limit: int,
                   max_history: int, seed: int, start=None, **metadata) -> list[dict]:
    """Up to ``limit`` windows from one univariate series: random cutoffs for each term's
    horizon, and a history of 2, 4, 8 or 16 horizons (capped by the data). ``start`` is the
    series' first timestamp, when known."""
    target = np.asarray(values, dtype=np.float32)
    if target.ndim != 1 or len(target) < 12:
        return []
    try:
        regime, horizons = frequency_regime(frequency), term_horizons(frequency)
    except ValueError:
        return []
    rng = random.Random(stable_seed(seed, dataset, series_id))
    per_term = max(1, math.ceil(limit / len(horizons)))
    candidates = []
    for term, horizon in horizons:
        latest, earliest = len(target) - horizon, max(4, min(4 * horizon, max_history))
        if latest < earliest:
            continue
        span = range(earliest, latest + 1)
        cutoffs = list(span) if len(span) <= per_term else rng.sample(span, per_term)
        candidates += [(cutoff, horizon, term) for cutoff in cutoffs]
    rng.shuffle(candidates)
    windows = []
    for cutoff, horizon, term in candidates:
        local = random.Random(stable_seed(seed, dataset, series_id, cutoff, horizon))
        length = local.choice(sorted({min(cutoff, max_history, max(4, m * horizon)) for m in (2, 4, 8, 16)}))
        history, future = repair(target[cutoff - length:cutoff]), repair(target[cutoff:cutoff + horizon])
        if history is None or future is None or len(future) != horizon or not admissible(history, future):
            continue
        windows.append(dict(history=history.tolist(), future=future.tolist(), freq=str(frequency),
                            term=term, dataset=dataset, source_series=series_id, source_group=group,
                            cutoff=cutoff, history_start=_offset(start, frequency, cutoff - length),
                            frequency_regime=regime, **metadata))
        if len(windows) >= limit:
            break
    return windows


def _offset(start, frequency: str, steps: int) -> str:
    """The timestamp ``steps`` periods after ``start`` ('' when the start is unknown)."""
    if not start:
        return ""
    import pandas as pd

    try:
        return timestamp(pd.Period(str(start), freq=frequency) + steps)
    except (ValueError, TypeError):
        return ""


def channels(target, layout: str = "channels_first") -> list[np.ndarray]:
    target = np.asarray(target, dtype=np.float32)
    if target.ndim == 1:
        return [target]
    if target.ndim == 2:
        return list(target if layout == "channels_first" else target.T)
    raise ValueError(f"unsupported target shape {target.shape}")


def clean(rows: list[dict], *, drop_constant: bool = True) -> list[dict]:
    """Remove protected histories, constant histories and duplicate windows."""
    seen, result = set(), []
    for row in sorted(rows, key=lambda r: (r["dataset"], str(r["source_series"]), r["cutoff"], len(r["future"]))):
        if history_digest(row["history"]) in EXCLUDED_HISTORIES:
            continue
        if drop_constant and np.all(np.asarray(row["history"]) == row["history"][0]):
            continue
        key = content_digest(row)
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


def interleave(rows: list[dict], keys: tuple[str, ...], seed: int, shuffle: bool = True) -> list[dict]:
    """Round-robin over the groups defined by ``keys`` (each group shuffled)."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(str(row.get(k)) for k in keys), []).append(row)
    if shuffle:
        for key, members in groups.items():
            random.Random(stable_seed(seed, *key)).shuffle(members)
    ordered = sorted(groups)
    depth = max(map(len, groups.values()), default=0)
    return [groups[k][i] for i in range(depth) for k in ordered if i < len(groups[k])]


def balanced(rows: list[dict], count: int, seed: int) -> list[dict]:
    """``count`` rows spread evenly over domains, then source families, frequencies and terms."""
    ordered = interleave(rows, ("domain", "source_family", "frequency_regime", "term"), seed)
    if len(ordered) < count:
        raise ValueError(f"only {len(ordered)} windows available for {count} requested")
    return interleave(ordered, ("domain",), stable_seed(seed, "domains"), shuffle=False)[:count]


# ---------------------------------------------------------------- GIFT-Eval
def gift_eval(root: str | Path, data: str | Path):
    """The pinned GIFT-Eval checkout's ``Dataset`` class, its 97 (dataset, term) cells and properties."""
    root = Path(root).resolve()
    if not (root / "src/gift_eval/data.py").is_file():
        raise FileNotFoundError(f"GIFT-Eval checkout not found at {root}; run scripts/setup.sh")
    os.environ["GIFT_EVAL"] = str(Path(data).resolve())
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    import yaml
    from gift_eval.data import Dataset

    names = [item["name"] for item in
             yaml.safe_load((root / "cli/conf/analysis/datasets/all_datasets.yaml").read_text())["datasets"]]
    cells = [(name, term) for name in names for term in ("short", "medium", "long")
             if term == "short" or name in LONG_TERMS]
    properties = json.loads((root / "notebooks/dataset_properties.json").read_text())
    return Dataset, cells, properties


def gift_dataset(Dataset, name: str, term: str):
    """A GIFT-Eval dataset, converted to univariate series when multivariate."""
    probe = Dataset(name=name, term=term, to_univariate=False)
    return (probe if probe.target_dim == 1 else Dataset(name=name, term=term, to_univariate=True)), probe.target_dim


def gift_metadata(properties: dict, name: str, variates: int) -> dict:
    key = name.split("/")[0].lower()
    meta = properties.get(PROPERTY_ALIASES.get(key, key), {})
    return dict(domain=str(meta.get("domain", "unknown")), num_variates=int(meta.get("num_variates", variates)))


def timestamp(value) -> str:
    if hasattr(value, "to_timestamp"):
        value = value.to_timestamp()
    return value.isoformat() if hasattr(value, "isoformat") else str(value)
