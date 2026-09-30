"""The released Toto-2.0 Family-and-Friends (FnF) ensemble and its blend with the router pool."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import numpy as np

FNF_REPO = "Datadog/Toto-2.0-Family-and-Friends"
FNF_REVISION = "bef47c99602b4993dd5aa3a2421cb685a2b96df7"
FNF_FILES = ("models.json", "feature_columns.json", "categories.json", "booster_manifest.json")
GIFT_FILES = ("test_features/*", "test_predictions/*")  # the released GIFT-Eval forecasts (about 17 GB)
FNF_MEMBERS = ("chronos-2", "timesfm-2.5", "flowstate", "tirex", "patchtst-fm", "toto-2.0-4m",
               "toto-2.0-22m", "toto-2.0-313m", "toto-2.0-1b", "toto-2.0-2.5b")


def canonical_frequency(value: str) -> str:
    """Express a pandas frequency in the vocabulary the FnF bundle uses (e.g. '1h' -> 'H')."""
    match = re.fullmatch(r"(\d*)([A-Za-z]+)(.*)", str(value))
    if match is None:
        raise ValueError(f"invalid forecast frequency: {value}")
    number, unit, suffix = match.groups()
    unit = {"min": "T", "ME": "M", "MS": "M", "QE": "Q", "YE": "A", "Y": "A"}.get(unit, unit.upper())
    return ("" if number == "1" else number) + unit + suffix.upper()


def ensure_fnf(root: str | Path | None = None, *, gift: bool = False) -> Path:
    """Download the released FnF gating models (once) and unpack one booster file per bucket;
    ``gift`` also downloads the bundle's member forecasts and features for the GIFT-Eval test set."""
    from huggingface_hub import snapshot_download

    root = Path(root) if root else None
    patterns = list(FNF_FILES) + (list(GIFT_FILES) if gift else [])
    path = Path(snapshot_download(FNF_REPO, revision=FNF_REVISION, allow_patterns=patterns, local_dir=root))
    boosters = path / "boosters"
    if not boosters.is_dir():
        import xgboost as xgb

        temporary = path / "boosters.tmp"
        temporary.mkdir(exist_ok=True)
        for key, value in json.loads((path / "booster_manifest.json").read_text()).items():
            booster = xgb.Booster()
            booster.load_model(bytearray(base64.b64decode(value)))
            booster.save_model(str(temporary / f"{key}.ubj"))
        temporary.rename(boosters)
    return path


class TotoFnF:
    """XGBoost gating over the ten member forecasts, per frequency/horizon bucket."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if tuple(json.loads((self.root / "models.json").read_text())) != FNF_MEMBERS:
            raise RuntimeError("unexpected Toto-FnF member roster")
        self.columns = tuple(map(str, json.loads((self.root / "feature_columns.json").read_text())))
        self.categories = json.loads((self.root / "categories.json").read_text())
        excluded = {"seasonality", "prediction_length", "num_variates", "freq", "domain"}
        self.series_features = tuple(c for c in self.columns if c not in excluded)
        self._boosters: dict[str, object] = {}

    def supports(self, frequency: str, term: str) -> bool:
        frequency = canonical_frequency(frequency).split("-", 1)[0]
        if frequency not in {str(v).split("-", 1)[0] for v in self.categories["freq"]}:
            return False
        return term == "short" or (frequency in {"10S", "10T", "15T", "5T", "H"} and term in {"medium", "long"})

    def _booster(self, frequency: str, term: str):
        import xgboost as xgb

        key = f"{canonical_frequency(frequency).split('-', 1)[0]}|{term}"
        if key not in self._boosters:
            booster = xgb.Booster()
            extracted = self.root / "boosters" / f"{key}.ubj"
            if extracted.is_file():
                booster.load_model(str(extracted))
            else:
                manifest = json.loads((self.root / "booster_manifest.json").read_text())
                booster.load_model(bytearray(base64.b64decode(manifest[key])))
            self._boosters[key] = booster
        return self._boosters[key]

    def feature_record(self, history, *, frequency: str, horizon: int, domain=None, num_variates: int = 1) -> dict:
        """The released feature values for one observed series: tsfeatures over its recent history."""
        import pandas as pd
        from gluonts.time_feature import get_seasonality
        from tsfeatures.tsfeatures import _get_feats  # tsfeatures() for one series, without a process pool

        frequency = canonical_frequency(frequency)
        seasonality = int(get_seasonality(frequency))
        values = np.asarray(history, dtype=np.float64).reshape(-1)[-max(256, 8 * seasonality):]
        frame = _get_feats(0, pd.DataFrame(dict(y=values)), freq=seasonality)
        record = {name: float(frame[name].iloc[0]) if name in frame else np.nan for name in self.series_features}
        record.update(seasonality=seasonality, prediction_length=horizon, num_variates=num_variates,
                      freq=frequency, domain=domain)
        return record

    def frame(self, records: list[dict]):
        """Typed XGBoost input rows (released column order and categories) for feature records."""
        import pandas as pd

        frame = pd.DataFrame.from_records(records).reindex(columns=self.columns)
        for name in self.columns:
            if name in ("freq", "domain"):
                frame[name] = pd.Categorical(frame[name], categories=self.categories[name])
            else:
                frame[name] = frame[name].astype(np.float32)
        return frame

    def features(self, history, *, frequency: str, horizon: int, domain=None, num_variates: int = 1):
        return self.frame([self.feature_record(history, frequency=frequency, horizon=horizon, domain=domain,
                                               num_variates=num_variates)])

    def weights(self, features, *, frequency: str, term: str) -> np.ndarray:
        """Gating weights [rows, 10] of the bucket's XGBoost softmax."""
        import xgboost as xgb

        logits = self._booster(frequency, term).predict(
            xgb.DMatrix(features, enable_categorical=True), output_margin=True).reshape(len(features), -1)
        logits = logits - logits.max(axis=1, keepdims=True)
        return np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

    def forecast(self, features, members: dict[str, np.ndarray], *, frequency: str, term: str) -> np.ndarray:
        """Gate member forecasts [rows, 9, horizon] by the bucket's XGBoost softmax."""
        weights = self.weights(features, frequency=frequency, term=term)
        available = [i for i, name in enumerate(FNF_MEMBERS) if name in members]
        weights = weights[:, available]
        weights /= weights.sum(axis=1, keepdims=True)
        stacked = np.stack([np.asarray(members[FNF_MEMBERS[i]]) for i in available], axis=1)
        return (weights[:, :, None, None] * stacked).sum(axis=1)

    def released(self, cell: str) -> np.ndarray:
        """The released FnF forecast [windows, horizon, 9] for a GIFT-Eval test cell ("dataset/freq/term"):
        the bundle's member forecasts gated with its features and dataset metadata (``ensure_fnf(gift=True)``)."""
        import pandas as pd

        dataset, frequency, term = cell.rsplit("/", 2)
        directory = f"{dataset}_{frequency}_{term}"
        with np.load(self.root / "test_features" / directory / "test_features.npz") as payload:
            frame = pd.DataFrame(payload["X"], columns=[str(n) for n in payload["feature_names"]])
        with np.load(self.root / "test_features" / directory / "test_metadata.npz") as payload:
            metadata = {name: payload[name].item() for name in payload.files}
        frame = frame.reindex(columns=self.series_features).astype(np.float32)
        for name in ("seasonality", "prediction_length", "num_variates"):
            frame[name] = np.float32(int(metadata[name]))
        for name in ("freq", "domain"):
            frame[name] = pd.Categorical([str(metadata[name])] * len(frame), categories=self.categories[name])
        members = {}
        for member in FNF_MEMBERS:
            with np.load(self.root / "test_predictions" / member / directory / "test_predictions.npz") as payload:
                members[member] = payload["predictions"].astype(np.float32)
        return self.forecast(frame.reindex(columns=self.columns), members, frequency=frequency,
                             term=term).transpose(0, 2, 1)


def blend(left: np.ndarray, right: np.ndarray, right_mass: float) -> np.ndarray:
    """q0 = (1 - a) q_router + a q_FnF, quantile by quantile; the router pool alone if FnF is missing."""
    left = np.asarray(left, dtype=np.float32)
    right = np.maximum.accumulate(np.asarray(right, dtype=np.float32), axis=-1)
    valid = np.isfinite(right).all(axis=-1)
    right = np.where(valid[..., None], right, left)
    return ((1.0 - right_mass) * left.astype(np.float64) + right_mass * right.astype(np.float64)).astype(np.float32)


def row_optimum(left, right, target, weight=None) -> float | None:
    """The blend weight in [0, 1] minimizing one row's weighted absolute error (None if uninformative)."""
    left, right, target = (np.asarray(v, dtype=np.float64).reshape(-1) for v in (left, right, target))
    weight = np.ones_like(left) if weight is None else np.asarray(weight, dtype=np.float64).reshape(-1)
    mask = np.isfinite(left) & np.isfinite(right) & np.isfinite(target)
    delta = right[mask] - left[mask]
    moving = delta != 0
    if not moving.any():
        return None
    roots = np.clip((target[mask][moving] - left[mask][moving]) / delta[moving], 0, 1)
    mass = np.abs(delta[moving]) * weight[mask][moving]
    if mass.sum() <= 0:
        return None
    order = np.argsort(roots, kind="stable")
    roots, cumulative = roots[order], np.cumsum(mass[order])
    lower = int(np.searchsorted(cumulative, cumulative[-1] / 2, side="left"))
    upper = int(np.searchsorted(cumulative, cumulative[-1] / 2, side="right"))
    return float(0.5 * (roots[lower] + roots[min(upper, len(roots) - 1)]))


def fit_blend_mass(left, right, target, weight=None, valid=None) -> float:
    """Mean of the per-row optimal FnF weights over rows where FnF is available ([rows, knots])."""
    rows = len(left)
    valid = np.ones(rows, bool) if valid is None else np.asarray(valid, bool)
    optima = [row_optimum(left[i], right[i], target[i], None if weight is None else weight[i])
              for i in range(rows) if valid[i]]
    optima = [value for value in optima if value is not None]
    if not optima:
        raise ValueError("no informative rows for the FnF blend weight")
    return float(np.mean(optima))
