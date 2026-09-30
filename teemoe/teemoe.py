"""The composed TeeMoE model: routing, numerical forecasting, native forecasting and analysis."""

from __future__ import annotations

import json
import tempfile
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import checkpoint, prompts
from .generation import generate_transformers, generate_vllm
from .model.backbone import decoder, load_base_model, load_tokenizer
from .model.controller import AGGREGATION, NUMERICAL_THRESHOLD, Controller, request_states
from .model.editor import AggregationEditor
from .model.mixture import attach_experts, expert_weights


LEVELS = np.arange(1, 10) / 10


@dataclass
class Forecast:
    output: str                      # "numerical" (aggregation expert) or "text" (native forecasting)
    timestamps: list[str]
    weights: np.ndarray              # expert mixture: aggregation, native, analysis
    quantiles: np.ndarray            # [horizon, 9] deciles 0.1 ... 0.9
    samples: np.ndarray | None = None  # [samples, horizon] generated trajectories (text output)

    @property
    def median(self) -> np.ndarray:
        return self.quantiles[:, 4]

    def trajectories(self, count: int = 25, seed: int = 0) -> np.ndarray:
        """Sample paths: the generated ones, or independent draws from the quantile forecast."""
        return self.samples if self.samples is not None else inverse_cdf(self.quantiles, count, seed)


def sample_quantiles(samples: np.ndarray) -> np.ndarray:
    """Deciles [horizon, 9] of sampled trajectories [samples, horizon] (order statistics)."""
    order = [int(np.round((len(samples) - 1) * q)) for q in LEVELS]
    return np.sort(np.asarray(samples, dtype=np.float64), axis=0)[order].T


def inverse_cdf(quantiles: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Independent draws per step from the piecewise-linear CDF through the deciles (constant tails)."""
    q = np.sort(np.asarray(quantiles, dtype=np.float64), axis=-1)
    uniforms = np.random.default_rng(seed).random((count, len(q)))
    return np.stack([np.interp(uniforms[:, t], LEVELS, q[t]) for t in range(len(q))], axis=1)


@dataclass
class Answer:
    text: str
    weights: np.ndarray
    choice: int | None = None        # index into the options, for multiple-choice questions


class TeeMoE:
    def __init__(self, root: str | Path, *, device: str = "cuda:0", backend: str = "vllm",
                 vllm_devices: str | None = None, vllm_python: str | None = None,
                 forecast_devices: tuple[str, ...] | None = None, environments: dict[str, str] | None = None,
                 granite: str | None = None, fnf_root: str | None = None) -> None:
        if backend not in {"vllm", "transformers"}:
            raise ValueError("backend must be 'vllm' or 'transformers'")
        self.root = Path(root)
        self.config = checkpoint.read_config(self.root)
        self.device, self.backend = torch.device(device), backend
        self.vllm_devices, self.vllm_python = vllm_devices, vllm_python
        self.forecast_devices = forecast_devices or (str(self.device),)
        self.environments, self.granite, self.fnf_root = environments or {}, granite, fnf_root
        self.tokenizer = load_tokenizer(self.config["base_model"], self.config["base_revision"])
        self.model = load_base_model(self.config["base_model"], self.config["base_revision"], device=self.device)
        self.adapters = checkpoint.adapter_dirs(self.root)
        attach_experts(self.model, self.adapters)
        hidden = self.model.config.get_text_config().hidden_size
        self.controller = Controller(hidden).to(self.device)
        self.controller.load_state_dict(checkpoint.load_state(self.root / "controller.safetensors"))
        self.editor = AggregationEditor(hidden).to(self.device)
        self.editor.load_state_dict(checkpoint.load_state(self.root / "editor.safetensors"))
        self._ensemble = None
        self._routing_cache = OrderedDict()

    @classmethod
    def from_pretrained(cls, path_or_repo: str | Path = checkpoint.DEFAULT_REPO, revision: str | None = None,
                        **options) -> TeeMoE:
        return cls(checkpoint.resolve(path_or_repo, revision), **options)

    # ------------------------------------------------------------------ routing
    @torch.no_grad()
    def route(self, chat_prompts: list[str]) -> np.ndarray:
        """Expert weights [requests, 3] (aggregation, native, analysis) for chat-formatted prompts."""
        balance = float(self.controller.signal_fraction)  # views with zero weight are skipped
        text, full = request_states(self.model, self.tokenizer, chat_prompts, text=balance < 1, full=balance > 0,
                                    cache=self._routing_cache)
        return self.controller(text, full).cpu().numpy()

    def numerical_output(self, weights: np.ndarray) -> np.ndarray:
        """Which forecasting requests the numerical decoder answers (the rest generate text)."""
        return weights[:, AGGREGATION] > NUMERICAL_THRESHOLD

    def execution_weights(self, weights: np.ndarray) -> np.ndarray:
        """The expert weights a request is run with: the routed ones (overridden for ablations)."""
        return weights

    # -------------------------------------------------------------- forecasting
    def forecast(self, history, *, horizon: int, frequency: str, start=None, context: str = "",
                 samples: int = 25, **options) -> Forecast:
        """Forecast one series; see :meth:`forecast_batch` for the request fields."""
        return self.forecast_batch([dict(history=history, horizon=horizon, frequency=frequency, start=start,
                                         context=context, **options)], samples=samples)[0]

    def forecast_batch(self, requests: list[dict], *, samples: int = 25, seed: int = 0,
                       editor_batch: int = 16) -> list[Forecast]:
        """Each request: ``history``, ``horizon``, ``frequency`` (pandas alias), and optionally
        ``start`` (first history timestamp), ``context`` (text), ``term`` (short/medium/long),
        ``domain``, ``dataset`` (for forecasting models with dataset-specific settings), explicit
        ``past_timestamps`` / ``future_timestamps``, a ready ``message`` (user message), a ready
        Toto-FnF forecast ``fnf_forecast`` [horizon, 9] and a sampling ``seed``."""
        rows = [self._forecast_row(request) for request in requests]
        chat = [prompts.chat_prompt(self.tokenizer, row["message"]) for row in rows]
        weights = self.route(chat)
        execution = self.execution_weights(weights)
        results: list[Forecast | None] = [None] * len(rows)
        numerical = np.flatnonzero(self.numerical_output(weights)).tolist()
        if numerical:
            quantiles = self._numerical([rows[i] for i in numerical], execution[numerical], editor_batch)
            for i, q in zip(numerical, quantiles, strict=True):
                results[i] = Forecast("numerical", rows[i]["future_timestamps"], weights[i], q)
        text = [i for i in range(len(rows)) if results[i] is None]
        if text:
            jobs = [dict(prompt=chat[i], weights=execution[i].tolist(), samples=samples, temperature=1.0,
                         max_tokens=prompts.forecast_max_tokens(len(rows[i]["future_timestamps"])),
                         regex=prompts.forecast_regex(rows[i]["future_timestamps"]),
                         seed=int(requests[i].get("seed", seed + i))) for i in text]
            for i, completions in zip(text, self.generate(jobs), strict=True):
                stamps = rows[i]["future_timestamps"]
                values = np.array([prompts.parse_forecast(c, stamps) for c in completions], dtype=np.float64)
                results[i] = Forecast("text", stamps, weights[i], sample_quantiles(values), values)
        return results

    def _forecast_row(self, request: dict) -> dict:
        history = [float(v) for v in request["history"]]
        horizon, frequency = int(request["horizon"]), request["frequency"]
        start = request.get("start") or "2000-01-03"
        past, future = request.get("past_timestamps"), request.get("future_timestamps")
        if past is None or future is None:
            past, future = prompts.history_timestamps(start, frequency, len(history), horizon)
        context = request.get("context", "") or ""
        message = request.get("message") or prompts.forecast_message(history, past, future, context)
        return dict(history=history, horizon=horizon, frequency=frequency, start=str(start),
                    past_timestamps=list(past), future_timestamps=list(future), message=message,
                    term=request.get("term", "short"), domain=request.get("domain"),
                    dataset=request.get("dataset", ""), fnf_forecast=request.get("fnf_forecast"))

    @contextmanager
    def _gpu_released(self, devices):
        """Move the backbone to CPU while other processes use its GPU."""
        shared = self.device.type == "cuda" and getattr(self, "model", None) is not None and any(
            str(d).split(":")[-1] == str(self.device.index or 0) for d in devices)
        if shared:
            self.model.to("cpu")
            torch.cuda.empty_cache()
        try:
            yield
        finally:
            if shared:
                self.model.to(self.device)

    def ensemble(self, rows: list[dict]) -> dict[str, list[np.ndarray]]:
        """Run the forecasting models and the numerical ensemble on forecasting requests; see
        :meth:`teemoe.ensemble.NumericalEnsemble.components`."""
        from .ensemble import CORE_ORDER, NumericalEnsemble
        from .ensemble.fnf import FNF_MEMBERS
        from .ensemble.forecasters import EXTRA_CANDIDATES, load, run_all

        if self._ensemble is None:
            self._ensemble = NumericalEnsemble(self.root / "router.ubj", self.config["fnf_mass"], self.fnf_root)
        # with ready FnF forecasts only the extra candidates of the FnF members are needed
        members = EXTRA_CANDIDATES if all(r.get("fnf_forecast") is not None for r in rows) else FNF_MEMBERS
        with tempfile.TemporaryDirectory(prefix="teemoe-fm-") as work:
            path = Path(work) / "requests.jsonl"
            keys = ("history", "horizon", "frequency", "start", "domain", "dataset")
            path.write_text("".join(json.dumps({k: r[k] for k in keys}) + "\n" for r in rows))
            with self._gpu_released(self.forecast_devices):
                run_all(path, Path(work) / "forecasts", devices=self.forecast_devices,
                        environments=self.environments, granite=self.granite,
                        models=[f"core/{n}" for n in CORE_ORDER] + [f"fnf/{n}" for n in members])
            core = {n: list(load(Path(work) / "forecasts" / f"core__{n}")) for n in CORE_ORDER}
            fnf = {n: list(load(Path(work) / "forecasts" / f"fnf__{n}")) for n in members}
        return self._ensemble.components(rows, core, fnf)

    @torch.no_grad()
    def _numerical(self, rows: list[dict], weights: np.ndarray, batch: int = 16) -> list[np.ndarray]:
        """The aggregation expert's forecasts: the reference refined with the 13 candidates."""
        parts = self.ensemble(rows)
        candidates, references = parts["candidates"], parts["reference"]
        results: list[np.ndarray | None] = [None] * len(rows)
        by_horizon: dict[int, list[int]] = {}
        for i, row in enumerate(rows):
            by_horizon.setdefault(int(row["horizon"]), []).append(i)
        for indices in by_horizon.values():  # one editor pass per batch of equal horizons
            for begin in range(0, len(indices), batch):
                chunk = indices[begin:begin + batch]
                tensor = lambda values: torch.as_tensor(np.stack([values[i] for i in chunk]), device=self.device)
                w = torch.as_tensor(weights[chunk], dtype=torch.float32, device=self.device)
                with expert_weights(self.model, w):
                    q = self.editor.predict(decoder(self.model), tensor(candidates), tensor(references))
                for i, value in zip(chunk, q.float().cpu().numpy(), strict=True):
                    results[i] = value
        return results

    # ----------------------------------------------------------------- analysis
    def analyze(self, series, question: str, options=None, **context) -> Answer:
        """Answer a question about one series (or a pair: ``series=[s1, s2]``)."""
        return self.analyze_batch([dict(series=series, question=question, options=options, **context)])[0]

    def analyze_batch(self, requests: list[dict], *, max_tokens: int = 64) -> list[Answer]:
        chat = []
        for r in requests:
            series = r["series"]
            series = [series] if np.ndim(series[0]) == 0 else series
            chat.append(prompts.chat_prompt(self.tokenizer, prompts.analysis_message(
                series, r["question"], r.get("options"), concept_definitions=r.get("concept_definitions", ""),
                clarification=r.get("clarification", ""))))
        weights = self.route(chat)
        jobs = [dict(prompt=p, weights=w.tolist(), samples=1, temperature=0.0, max_tokens=max_tokens, seed=1)
                for p, w in zip(chat, self.execution_weights(weights), strict=True)]
        answers = []
        for r, w, completions in zip(requests, weights, self.generate(jobs), strict=True):
            text = completions[0].strip()
            choice = prompts.parse_choice(text, r["options"]) if r.get("options") else None
            answers.append(Answer(text, w, choice))
        return answers

    # --------------------------------------------------------------- generation
    def generate(self, jobs: list[dict]) -> list[list[str]]:
        if self.backend == "transformers":
            return generate_transformers(self.model, self.tokenizer, jobs)
        devices = self.vllm_devices if self.vllm_devices is not None else str(self.device.index or 0)
        with self._gpu_released(str(devices).split(",")):
            return generate_vllm(jobs, adapters=self.adapters,
                                 model=self.config["base_model"], revision=self.config["base_revision"],
                                 devices=devices, python=self.vllm_python)
