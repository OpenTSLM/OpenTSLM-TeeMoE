"""Pretrained forecasting models behind the aggregation expert.

Eight *core* models feed the XGBoost router; the ten Toto-FnF *members* feed
the released FnF ensemble (five of them are also extra editor candidates).
Each model runs in a worker process so that models with conflicting
dependencies can use their own Python environments:

    python -m teemoe.ensemble.forecasters --model core/chronos2 --requests r.jsonl --output out/

Requests are JSON lines with ``history``, ``horizon``, ``frequency``, ``start``
(first history timestamp) and optionally ``domain``. A model's output directory
holds nine deciles per step, packed as ``quantiles.npy`` [sum(horizons), 9]
with row ``offsets.npy``; ``load`` reads it back.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .fnf import canonical_frequency
from ..runtime import worker_pool

LEVELS = [i / 10 for i in range(1, 10)]
GRANITE = dict(repository="https://github.com/ibm-granite/granite-tsfm.git",
               revision="31c7a6bd640c41e53777c0afe3a13d24e00f7fe7")

# name -> (kind, Hugging Face id, revision, options)
CORE_MODELS = {
    "tirex2": ("tirex2", "NX-AI/TiRex-2-gifteval-pretrain", "6f0fe5ec3247d55b53ddf78911e3bd52d8d7a4c3",
               {"batch_size": 128}),
    "toto2": ("toto2", "Datadog/Toto-2.0-2.5B", "51a2812bbe449437c01b79c0e425ed578f335f5b", {}),
    "chronos2": ("chronos2", "amazon/chronos-2", "29ec3766d36d6f73f0696f85560a422f50e8498c", {}),
    "timer_s1": ("timer_s1", "bytedance-research/Timer-S1", "8911430cc7f32add5c8913afe12e3b05742f5bb2",
                 {"max_context": 11520}),
    "timesfm25": ("timesfm25", "google/timesfm-2.5-200m-pytorch", "1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
                  {"max_context": 8192, "max_horizon": 1024, "per_core_batch_size": 64}),
    "moirai2": ("moirai2", "Salesforce/moirai-2.0-R-small", "30f43ff08c8494f4943ae1521e9d4e94a0fbb389",
                {"max_context": 4000}),
    "flowstate": ("flowstate", "ibm-research/flowstate", "ec815e416006b6a74130e1c79c6c67004c12c0c8", {}),
    "patchtst_fm": ("patchtst_fm", "ibm-research/patchtst-fm-r1", "67dc5f9a1d26bbc782b3ac45bcdb899ce957e681", {}),
}
FNF_MODELS = {
    "chronos-2": ("chronos2", "amazon/chronos-2", "29ec3766d36d6f73f0696f85560a422f50e8498c", {}),
    "timesfm-2.5": ("timesfm25", "google/timesfm-2.5-200m-pytorch", "1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
                    {"max_context": 15360, "max_horizon": 1024, "batch_size": 128}),
    "flowstate": ("flowstate", "ibm-research/flowstate", "ec815e416006b6a74130e1c79c6c67004c12c0c8",
                  {"batch_size": 16}),
    "tirex": ("tirex11", "NX-AI/TiRex-1.1-gifteval", "01b1ebc83a87e82b119f907fe7e43d015db23e9d",
              {"batch_size": 512}),
    "patchtst-fm": ("patchtst_fm", "ibm-research/patchtst-fm-r1", "67dc5f9a1d26bbc782b3ac45bcdb899ce957e681", {}),
    "toto-2.0-4m": ("toto2", "Datadog/Toto-2.0-4m", "8306a9801cf98c0f5ffe4b2dcc8f496e616d84d9", {}),
    "toto-2.0-22m": ("toto2", "Datadog/Toto-2.0-22m", "685e4ae3e2be8d8998025e53dd98e7fdcb296a89", {}),
    "toto-2.0-313m": ("toto2", "Datadog/Toto-2.0-313m", "a7bab288f5e95f8606f8306f86659357e1c001ef", {}),
    "toto-2.0-1b": ("toto2", "Datadog/Toto-2.0-1B", "1604e1a5242884fb9848f88c4ced14f4dc62d9d3", {}),
    "toto-2.0-2.5b": ("toto2", "Datadog/Toto-2.0-2.5B", "51a2812bbe449437c01b79c0e425ed578f335f5b", {}),
}
EXTRA_CANDIDATES = ("tirex", "toto-2.0-4m", "toto-2.0-22m", "toto-2.0-313m", "toto-2.0-1b")
ALL_MODELS = {**{f"core/{k}": v for k, v in CORE_MODELS.items()},
              **{f"fnf/{k}": v for k, v in FNF_MODELS.items()}}
MAX_HISTORY = 8192


def interpolate(values: torch.Tensor, source, target=LEVELS) -> torch.Tensor:
    source = torch.as_tensor(source, device=values.device, dtype=values.dtype)
    target = torch.as_tensor(target, device=values.device, dtype=values.dtype)
    upper = torch.searchsorted(source, target).clamp(1, source.numel() - 1)
    lower = upper - 1
    weight = ((target - source[lower]) / (source[upper] - source[lower])).clamp(0, 1)
    low, high = values.index_select(-1, lower), values.index_select(-1, upper)
    return low + weight * (high - low)


REPO_ROOT = Path(__file__).resolve().parents[2]


def child_gpus(devices) -> str:
    """CUDA_VISIBLE_DEVICES for a child process that should use this process's GPU(s) ``devices``
    (e.g. "cuda:1" or "0,1"): indices here are relative to our own CUDA_VISIBLE_DEVICES."""
    indices = [str(d).split(":")[-1] for d in str(devices).split(",")]
    if visible := os.environ.get("CUDA_VISIBLE_DEVICES"):
        indices = [visible.split(",")[int(i)] for i in indices]
    return ",".join(indices)


def default_environments() -> dict[str, str]:
    """The forecasting-model environments created by scripts/setup.sh, where present."""
    found = {}
    for key, name in (("default", ".venv-fm"), ("timer_s1", ".venv-timer"),
                      ("moirai2", ".venv-moirai"), ("tirex2", ".venv-tirex")):
        if (python := REPO_ROOT / name / "bin/python").exists():
            found[key] = str(python)
    return found


def use_granite(root: str | Path | None) -> None:
    """Import FlowState / PatchTST-FM from the pinned granite-tsfm checkout."""
    root = Path(root or os.environ.get("TEEMOE_GRANITE") or REPO_ROOT / "third_party/granite-tsfm").resolve()
    if not (root / "tsfm_public").is_dir():
        raise FileNotFoundError(f"granite-tsfm checkout not found at {root}; run scripts/setup.sh")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import transformers.utils as utils
    from urllib.parse import urlparse
    from transformers.utils import hub

    utils.__dict__.setdefault("is_offline_mode", hub.is_offline_mode)
    utils.__dict__.setdefault("is_remote_url", lambda v: urlparse(str(v)).scheme in {"http", "https"})
    utils.__dict__.setdefault("download_url", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("unsupported")))


def _tied(cls):
    if not hasattr(cls, "all_tied_weights_keys"):
        cls.all_tied_weights_keys = {}
    return cls


class Chronos2:
    def __init__(self, model_id, revision, device, **_):
        from chronos import Chronos2Pipeline
        self.pipeline = Chronos2Pipeline.from_pretrained(model_id, device_map=device, revision=revision)
        self.device = device

    def predict(self, histories, horizon, metadata):
        quantiles, _ = self.pipeline.predict_quantiles([{"target": h.cpu().numpy()} for h in histories],
                                                       prediction_length=horizon, quantile_levels=LEVELS)
        return interpolate(torch.stack([q.squeeze(0).float() for q in quantiles]).to(self.device), LEVELS)


class Toto2:
    def __init__(self, model_id, revision, device, **_):
        from toto2 import Toto2Model
        self.model = Toto2Model.from_pretrained(model_id, revision=revision).to(device).eval()
        self.device = device

    def predict(self, histories, horizon, metadata):
        length = max(h.numel() for h in histories)
        target = torch.zeros(len(histories), 1, length, device=self.device)
        mask = torch.zeros_like(target, dtype=torch.bool)
        for i, h in enumerate(histories):
            target[i, 0, -h.numel():], mask[i, 0, -h.numel():] = h, True
        pad = (-length) % int(self.model.config.patch_size)
        if pad:
            target = torch.nn.functional.pad(target, (pad, 0))
            mask = torch.nn.functional.pad(mask, (pad, 0), value=False)
        native = self.model.forecast(dict(target=target, target_mask=mask, series_ids=torch.zeros(
            len(histories), 1, dtype=torch.long, device=self.device)), horizon=horizon,
            decode_block_size=None, has_missing_values=not bool(mask.all()))
        return interpolate(native[:, :, 0].permute(1, 2, 0).float(), LEVELS)


class TiRex2:
    def __init__(self, model_id, revision, device, batch_size=128, **_):
        bin_dir = Path(sys.executable).parent
        if shutil.which("ninja") is None and (bin_dir / "ninja").is_file():
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
        from tirex2 import TimeseriesType, load_model
        if device.startswith("cuda:"):
            torch.cuda.set_device(int(device.split(":")[1]))
        self.pipeline = load_model(model_id, device="cuda" if device.startswith("cuda") else "cpu",
                                   hf_kwargs={"revision": revision})
        self.series, self.device, self.batch_size = TimeseriesType, device, batch_size

    def predict(self, histories, horizon, metadata):
        series = [self.series(target=h.cpu().unsqueeze(0), past_covariates=None, future_covariates=None)
                  for h in histories]
        out = self.pipeline.forecast(series, prediction_length=horizon, output_type="torch",
                                     batch_size=self.batch_size)
        native = torch.stack([v[0].transpose(0, 1).float() for v in out]).to(self.device)
        return interpolate(native, self.pipeline._quantile_levels())


class Moirai2:
    def __init__(self, model_id, revision, device, max_context=4000, **_):
        from huggingface_hub import snapshot_download
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        self.model = Moirai2Forecast(
            module=Moirai2Module.from_pretrained(snapshot_download(model_id, revision=revision)),
            prediction_length=1, context_length=max_context, target_dim=1, feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0).to(device).eval()
        self.device, self.max_context = device, max_context

    def predict(self, histories, horizon, metadata):
        self.model.hparams.prediction_length = horizon
        out = self.model.predict([h[-self.max_context:].cpu().numpy() for h in histories])
        native = torch.as_tensor(out, dtype=torch.float32, device=self.device).transpose(1, 2)
        return interpolate(torch.cummax(native, dim=-1).values, LEVELS)


class TimerS1:
    def __init__(self, model_id, revision, device, max_context=11520, **_):
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, trust_remote_code=True,
                                                          device_map={"": torch.device(device)},
                                                          dtype=torch.bfloat16).eval()
        self.device, self.max_context = device, max_context

    def predict(self, histories, horizon, metadata):
        out = torch.zeros(len(histories), horizon, 9, device=self.device)
        groups = defaultdict(list)
        for i, h in enumerate(histories):
            groups[min(h.numel(), self.max_context)].append(i)
        for indices in groups.values():
            values = [histories[i][-self.max_context:].to(self.device) for i in indices]
            peak = torch.stack([v.abs().amax() for v in values])
            scale = torch.where(peak > 1e12, peak / 1e6, torch.ones_like(peak))
            context = torch.stack([v / s for v, s in zip(values, scale)]).to(torch.bfloat16)
            with torch.autocast(device_type=context.device.type, dtype=torch.bfloat16, enabled=context.is_cuda):
                q = self.model.generate(context, max_new_tokens=horizon, revin=True)
            q = torch.cummax(q.transpose(1, 2).float() * scale.view(-1, 1, 1), dim=-1).values
            out[indices] = q
        return interpolate(out, LEVELS)


class TimesFM25:
    def __init__(self, model_id, revision, device, max_context=8192, max_horizon=1024,
                 per_core_batch_size=64, batch_size=None, **_):
        import timesfm
        from huggingface_hub import hf_hub_download
        pipeline = timesfm.TimesFM_2p5_200M_torch(torch_compile=False)
        pipeline.model.device, pipeline.model.device_count = torch.device(device), 1
        pipeline.load_checkpoint(hf_hub_download(model_id, pipeline.WEIGHTS_FILENAME, revision=revision),
                                 torch_compile=False)
        pipeline.compile(timesfm.ForecastConfig(
            max_context=max_context, max_horizon=max_horizon,
            per_core_batch_size=batch_size or per_core_batch_size, normalize_inputs=True,
            use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=True,
            fix_quantile_crossing=True))
        self.pipeline, self.device = pipeline, device

    def predict(self, histories, horizon, metadata):
        _, native = self.pipeline.forecast(horizon=horizon, inputs=[h.cpu().numpy() for h in histories])
        return interpolate(torch.as_tensor(native[..., 1:], device=self.device).float(), LEVELS)


def fills_context(model, length: int, horizon: int) -> bool:
    """Whether PatchTST-FM truncates an input to its context rather than padding it. Its code cannot
    batch the two cases together; it forecasts at least d_patch * max(pretrain_mask_cont, 2) steps."""
    config = model.config
    forecast = max(horizon, config.d_patch * max(config.pretrain_mask_cont, 2))
    return length + forecast >= config.context_length


class PatchTSTFM:
    def __init__(self, model_id, revision, device, granite=None, **_):
        use_granite(granite)
        from tsfm_public import PatchTSTFMForPrediction
        self.model = _tied(PatchTSTFMForPrediction).from_pretrained(model_id, revision=revision).to(device).eval()

    def predict(self, histories, horizon, metadata):
        out = [None] * len(histories)
        for fills in (True, False):  # truncated and padded inputs are forecast separately
            indices = [i for i, h in enumerate(histories) if fills_context(self.model, h.numel(), horizon) == fills]
            if indices:
                result = self.model(past_values=[histories[i] for i in indices], prediction_length=horizon,
                                    quantile_levels=LEVELS)
                for i, q in zip(indices, result.quantile_outputs, strict=True):
                    out[i] = q.squeeze(-1).transpose(0, 1).float()
        return torch.stack(out)


class FlowState:
    """FlowState with its authors' frequency/domain time-scale factor."""

    def __init__(self, model_id, revision, device, granite=None, **_):
        use_granite(granite)
        from tsfm_public import FlowStateForPrediction
        self.model = _tied(FlowStateForPrediction).from_pretrained(model_id, revision=revision).to(device).eval()
        self.device = device
        self.levels = list(self.model.config.quantiles)
        self.context = int(self.model.config.context_length)

    def factor(self, history, meta):
        frequency = str(meta.get("frequency") or "H").strip().upper()
        domain = meta.get("domain") or None
        if frequency.endswith("D") and "WED" not in frequency and domain is None:
            values = history.float() - history.float().mean()
            variance = values.square().mean().clamp_min(1e-8)
            weekly = (values[7:] * values[:-7]).mean() / variance if values.numel() >= 15 else -1
            annual = ((values[365:] * values[:-365]).mean() / variance) if values.numel() >= 366 else 0.5
            domain = "Transport" if values.numel() >= 15 and weekly > annual else None
        if frequency == "S":
            return 24.0 / 3600.0
        from tsfm_public.models.flowstate.utils.utils import get_fixed_factor
        weekly = meta.get("dataset", "").startswith("bizitobs_l2c")
        return float(get_fixed_factor(frequency, domain)) / (7 if weekly else 1)

    def predict(self, histories, horizon, metadata):
        out = torch.zeros(len(histories), horizon, 9, device=self.device)
        groups = defaultdict(list)
        clipped = []
        for i, h in enumerate(histories):
            factor = self.factor(h, metadata[i])
            clipped.append(h[-max(1, int(self.context / factor)):].to(self.device))
            groups[(factor, clipped[-1].numel())].append(i)
        median = self.levels.index(0.5)
        for (factor, _), indices in groups.items():
            context = torch.stack([clipped[i] for i in indices], dim=1).unsqueeze(-1)
            pieces, remaining = [], horizon
            while remaining > 0:
                result = self.model(past_values=context, scale_factor=factor, prediction_length=remaining,
                                    batch_first=False)
                q = getattr(result, "quantile_outputs", None)
                q = (result.prediction_outputs if q is None else q).squeeze(-1).transpose(1, 2).float()
                piece = q[:, :min(remaining, q.shape[1])]
                pieces.append(piece)
                remaining -= piece.shape[1]
                if remaining:
                    context = torch.cat((context, piece[..., median].transpose(0, 1).unsqueeze(-1)), 0)
                    context = context[-max(1, int(self.context / factor)):]
            forecast = torch.cat(pieces, dim=1)
            for local, i in enumerate(indices):
                out[i] = forecast[local].clamp_min(0) if bool((clipped[i] >= 0).all()) else forecast[local]
        return out


class FnFMember:
    """A Toto-FnF member with the settings of the released FnF bundle."""

    def __init__(self, kind, model_id, revision, device, granite=None, **options):
        self.kind, self.device, self.options = kind, device, options
        if kind == "chronos2":
            from chronos import BaseChronosPipeline
            self.model = BaseChronosPipeline.from_pretrained(model_id, revision=revision,
                                                             torch_dtype=torch.float32, device_map=device)
        elif kind == "timesfm25":
            from huggingface_hub import hf_hub_download
            from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
            self.model = TimesFM_2p5_200M_torch(torch_compile=False)
            self.model.model.device, self.model.model.device_count = torch.device(device), 1
            self.model.load_checkpoint(hf_hub_download(model_id, TimesFM_2p5_200M_torch.WEIGHTS_FILENAME,
                                                       revision=revision), torch_compile=False)
        elif kind == "tirex11":
            from tirex import load_model
            self.model = load_model(model_id, device=device, backend="torch", hf_kwargs={"revision": revision})
        elif kind == "toto2":
            from toto2 import Toto2Model
            self.model = Toto2Model.from_pretrained(model_id, revision=revision, map_location=device).to(device).eval()
        elif kind in {"flowstate", "patchtst_fm"}:
            use_granite(granite)
            from tsfm_public import FlowStateForPrediction, PatchTSTFMForPrediction
            cls = _tied(FlowStateForPrediction if kind == "flowstate" else PatchTSTFMForPrediction)
            if kind == "flowstate":
                self.model = cls.from_pretrained(model_id, revision=revision).to(device)
                self.model.config.min_context, self.model.config.device = 0, device
            else:
                self.model = cls.from_pretrained(model_id, revision=revision, device_map=device).eval()
        else:
            raise ValueError(f"unknown FnF member kind {kind}")

    def predict(self, histories: list[np.ndarray], horizon: int, frequency: str, starts: list) -> np.ndarray:
        """[rows, horizon, 9] sorted deciles for one batch with a common horizon and frequency."""
        batch = self.options.get("batch_size", 128)
        if self.kind == "chronos2":
            out, _ = self.model.predict_quantiles([dict(target=h) for h in histories], prediction_length=horizon,
                                                  quantile_levels=LEVELS, predict_batches_jointly=False)
            q = np.stack([np.asarray(v, np.float32).reshape(horizon, 9) for v in out])
        elif self.kind == "timesfm25":
            from timesfm.configs import ForecastConfig
            contexts, scales = [], []
            for h in histories:  # rescale only values large enough to overflow float32 statistics
                peak = float(np.max(np.abs(h)))
                large = peak > np.sqrt(float(np.finfo(np.float32).max) / len(h)) / 16.0
                contexts.append(np.asarray(h, np.float64) / (peak if large else 1.0))
                scales.append(peak if large else 1.0)
            patch = int(self.model.model.p)
            self.model.compile(forecast_config=ForecastConfig(
                max_context=min(self.options.get("max_context", 15360), -(-max(map(len, contexts)) // patch) * patch),
                max_horizon=self.options.get("max_horizon", 1024), infer_is_positive=True,
                use_continuous_quantile_head=True, fix_quantile_crossing=True, force_flip_invariance=True,
                return_backcast=False, normalize_inputs=True, per_core_batch_size=batch))
            _, out = self.model.forecast(horizon=horizon, inputs=contexts)
            q = np.asarray(out, np.float64)[:, :horizon, 1:] * np.asarray(scales)[:, None, None]
        elif self.kind == "tirex11":
            out, _ = self.model.forecast([np.asarray(h, np.float32) for h in histories], prediction_length=horizon,
                                         output_type="numpy", batch_size=batch, resample_strategy=None)
            q = np.asarray(out, np.float32)
        elif self.kind == "patchtst_fm":  # as PatchTSTFM.predict: long and short inputs in separate calls
            q = [None] * len(histories)
            for fills in (True, False):
                indices = [i for i, h in enumerate(histories) if fills_context(self.model, len(h), horizon) == fills]
                if indices:
                    with torch.inference_mode():
                        out = self.model(past_values=[torch.as_tensor(histories[i], dtype=torch.float32,
                                                                      device=self.device) for i in indices],
                                         prediction_length=horizon, quantile_levels=LEVELS).quantile_outputs
                    for i, v in zip(indices, out, strict=True):
                        q[i] = np.asarray(v.squeeze(-1).float().cpu()).T
        else:  # FlowState and Toto through their GluonTS predictors
            import pandas as pd
            entries = [dict(target=np.asarray(h, np.float32), start=pd.Period(start, freq=frequency))
                       for h, start in zip(histories, starts, strict=True)]
            forecasts = list(self._gluonts_predictor(horizon, frequency).predict(entries))
            q = np.stack([np.stack([f.quantile(str(level)) for level in LEVELS], axis=-1) for f in forecasts])
        return np.sort(np.asarray(q, dtype=np.float32).reshape(len(histories), horizon, 9), axis=-1)

    def _gluonts_predictor(self, horizon, frequency):
        if self.kind == "flowstate":
            from notebooks.hfdemo.flowstate.gift_wrapper import FlowState_Gift_Wrapper
            return FlowState_Gift_Wrapper(self.model, horizon, n_ch=1, batch_size=self.options.get("batch_size", 16),
                                          f=frequency, device=self.device, domain=None)
        from toto2 import Toto2GluonTSModel, Toto2GluonTSModelConfig
        config = Toto2GluonTSModelConfig(prediction_length=horizon, context_length=4096, target_dim=1,
                                         past_feat_dynamic_real_dim=0)
        wrapper = Toto2GluonTSModel(self.model, config).to(self.device).eval()
        return wrapper.create_predictor(batch_size=self.options.get("batch_size", 1), device=self.device)


CORE_CLASSES = dict(chronos2=Chronos2, toto2=Toto2, tirex2=TiRex2, timer_s1=TimerS1, timesfm25=TimesFM25,
                    moirai2=Moirai2, flowstate=FlowState, patchtst_fm=PatchTSTFM)


def knots(horizon: int, count: int = 8) -> np.ndarray:
    """The ``count`` evenly spaced horizon steps at which training targets are compared."""
    return np.rint(np.linspace(0, horizon - 1, count)).astype(np.int64)


@torch.no_grad()
def run_model(name: str, requests: list[dict], device: str = "cuda:0", granite=None, batch_size: int = 32,
              at_knots: bool = False, log_every: int = 50) -> list[np.ndarray]:
    """Forecast every request with one model: sorted deciles [horizon, 9] per request (or only at
    the eight training knots). A batch the model cannot forecast is returned as NaN."""
    kind, model_id, revision, options = ALL_MODELS[name]
    member = name.startswith("fnf/")
    model = (FnFMember(kind, model_id, revision, device, granite=granite, **options) if member
             else CORE_CLASSES[kind](model_id, revision, device, granite=granite, **options))
    size = options.get("batch_size", batch_size) if member else batch_size
    groups = defaultdict(list)  # batches share a horizon and a frequency
    for i, r in enumerate(requests):
        groups[(int(r["horizon"]), canonical_frequency(r["frequency"]))].append(i)
    out: list[np.ndarray | None] = [None] * len(requests)
    for number, ((horizon, frequency), indices) in enumerate(sorted(groups.items())):
        for begin in range(0, len(indices), size):
            chunk = indices[begin:begin + size]
            histories = [np.asarray(requests[i]["history"], np.float32)[-MAX_HISTORY:] for i in chunk]
            try:
                if member:
                    q = model.predict(histories, horizon, frequency, [requests[i]["start"] for i in chunk])
                else:
                    meta = [dict(frequency=canonical_frequency(requests[i]["frequency"]),
                                 domain=requests[i].get("domain"), dataset=requests[i].get("dataset", ""))
                            for i in chunk]
                    q = model.predict([torch.as_tensor(h, device=device) for h in histories], horizon,
                                      meta).float().cpu().numpy()
                q = np.sort(q, axis=-1)
            except (RuntimeError, ValueError, NotImplementedError, KeyError) as error:
                print(json.dumps(dict(model=name, horizon=horizon, rows=len(chunk), error=str(error)[:200])),
                      flush=True)
                q = np.full((len(chunk), horizon, 9), np.nan, np.float32)
            q[~(np.isfinite(q).all(axis=(1, 2)) & (np.abs(q) < 1e30).all(axis=(1, 2)))] = np.nan
            for local, i in enumerate(chunk):
                out[i] = q[local][knots(horizon)] if at_knots else q[local]
        if log_every and number % log_every == 0:
            print(json.dumps(dict(model=name, groups=number + 1, of=len(groups))), flush=True)
    return out


class Forecasts:
    """One model's saved forecasts: ``forecasts[i]`` is request i's [horizon, 9] deciles (memory-mapped)."""

    def __init__(self, path: str | Path) -> None:
        self.offsets = np.load(Path(path) / "offsets.npy", mmap_mode="r")
        self.values = np.load(Path(path) / "quantiles.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, index: int) -> np.ndarray:
        return np.asarray(self.values[self.offsets[index]:self.offsets[index + 1]])


def save(path: str | Path, forecasts) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.mkdir(parents=True, exist_ok=True)
    offsets = np.concatenate(([0], np.cumsum([len(f) for f in forecasts]))).astype(np.int64)
    np.save(temporary / "offsets.npy", offsets)
    np.save(temporary / "quantiles.npy", np.concatenate(list(forecasts)).astype(np.float32))
    temporary.rename(path)


def load(path: str | Path) -> Forecasts:
    return Forecasts(path)


def read_requests(path: str | Path, shard: int = 0, shards: int = 1) -> list[dict]:
    """The ``shard``-th of ``shards`` contiguous parts of a requests JSONL file."""
    with Path(path).open("rb") as stream:
        lines = sum(1 for _ in stream)
    begin, end = lines * shard // shards, lines * (shard + 1) // shards
    with Path(path).open() as stream:
        return [json.loads(line) for i, line in enumerate(stream) if begin <= i < end]


def output_name(model: str, shard: int | None = None) -> str:
    return model.replace("/", "__") + ("" if shard is None else f".part{shard:03d}")


def run_all(requests_path: str | Path, output: str | Path, *, devices=("cuda:0",),
            environments: dict[str, str] | None = None, granite=None, models=None, shards: int = 1,
            at_knots=()) -> Path:
    """Run every model on every request, ``shards`` worker processes per model spread over
    ``devices``. ``environments`` maps a model name, model kind or "default" to the Python
    executable of the environment it runs in. Finished parts are reused."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    environments = environments or default_environments()
    models = [m for m in (models or ALL_MODELS) if not (output / output_name(m)).exists()]
    pending = [(m, k) for m in models for k in range(shards) if not (output / output_name(m, k)).exists()]
    running: list[tuple[subprocess.Popen, str]] = []
    free = list(devices)
    with worker_pool() as processes:
        while pending or running:
            while pending and free:
                (name, shard), device = pending.pop(0), free.pop(0)
                kind = ALL_MODELS[name][0]
                python = environments.get(name.split("/")[1], environments.get(kind, environments.get(
                    "default", sys.executable)))
                command = [python, "-m", "teemoe.ensemble.forecasters", "--model", name, "--requests",
                           str(requests_path), "--output", str(output / output_name(name, shard)),
                           "--shard", str(shard), "--shards", str(shards), "--device", "cuda:0"]
                command += ["--granite", str(granite)] if granite else []
                command += ["--knots"] if name in at_knots else []
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=child_gpus(device),
                           PYTHONPATH=os.pathsep.join([str(REPO_ROOT), os.environ.get("PYTHONPATH", "")]))
                process = subprocess.Popen(command, env=env, start_new_session=True)
                processes.append(process)
                running.append((process, device))
            time.sleep(1)
            for process, device in list(running):
                if process.poll() is not None:
                    running.remove((process, device))
                    free.append(device)
                    if process.returncode:
                        raise RuntimeError(f"forecaster worker failed: {' '.join(map(str, process.args))}")
    for name in models:  # join the parts
        parts = [load(output / output_name(name, k)) for k in range(shards)]
        destination = output / (output_name(name) + ".tmp")
        destination.mkdir(exist_ok=True)
        shift = np.cumsum([0] + [int(part.offsets[-1]) for part in parts])[:-1]
        np.save(destination / "offsets.npy", np.concatenate(
            [np.asarray(part.offsets[:-1]) + s for part, s in zip(parts, shift, strict=True)]
            + [[shift[-1] + parts[-1].offsets[-1]]]).astype(np.int64))
        np.save(destination / "quantiles.npy", np.concatenate([np.asarray(part.values) for part in parts]))
        destination.rename(output / output_name(name))
        for k in range(shards):
            shutil.rmtree(output / output_name(name, k))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(ALL_MODELS))
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="output directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--granite")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--knots", action="store_true", help="keep only the eight training knots")
    args = parser.parse_args()
    requests = read_requests(args.requests, args.shard, args.shards)
    forecasts = run_model(args.model, requests, device=args.device, granite=args.granite, at_knots=args.knots)
    save(args.output, forecasts)


if __name__ == "__main__":
    main()
