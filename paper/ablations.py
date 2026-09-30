"""Evaluate the paper's ablations and baselines with the regular benchmark scripts.

    python paper/ablations.py VARIANT BENCHMARK [evaluation options]

VARIANT
  expert:aggregation, expert:native, expert:analysis
                          one expert at full strength (numerical output only for the
                          aggregation expert on GIFT-Eval)
  top1                    only the controller's highest-weighted expert
  fixed:A,N,S             fixed expert weights, e.g. fixed:1,1,1 or fixed:0.333,0.333,0.333;
                          the controller still chooses between numerical and text output
  base                    Qwen3.6-27B without adapters
  ensemble:NAME           a numerical ensemble without the language model, NAME one of
                          reference (router pool + Toto-FnF), router (XGBoost-weighted core 8),
                          fnf (Toto-FnF), equal8 (equal-weight core 8), equal13 (all 13)
  joint                   the joint baseline trained by paper/joint.py (use its --checkpoint)
BENCHMARK  gift, cik or tse

For example:  python paper/ablations.py top1 cik --output results/ablations/top1_cik.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "paper")]

from teemoe import checkpoint  # noqa: E402
from teemoe.ensemble.router import pool_quantiles  # noqa: E402
from teemoe.model.backbone import load_tokenizer  # noqa: E402
from teemoe.model.controller import AGGREGATION  # noqa: E402
from teemoe.teemoe import TeeMoE  # noqa: E402

EXPERT_INDEX = dict(aggregation=0, native=1, analysis=2)


class Ablation(TeeMoE):
    def __init__(self, root, *, variant: str, benchmark: str, **options) -> None:
        self.kind, _, self.argument = variant.partition(":")
        self.benchmark = benchmark
        if self.kind == "ensemble":  # no language model: only the numerical ensemble
            self.root, self.config = Path(root), checkpoint.read_config(Path(root))
            self.device = torch.device(options.get("device", "cpu"))
            self.forecast_devices = options.get("forecast_devices") or (str(self.device),)
            self.environments, self.granite = options.get("environments") or {}, options.get("granite")
            self.fnf_root, self._ensemble = options.get("fnf_root"), None
            self.tokenizer = load_tokenizer(self.config["base_model"], self.config["base_revision"])
            return
        super().__init__(root, **options)

    def fixed(self) -> np.ndarray | None:
        if self.kind == "expert":
            return np.eye(3)[EXPERT_INDEX[self.argument]]
        if self.kind == "base":
            return np.zeros(3)
        if self.kind == "fixed":
            return np.asarray([float(v) for v in self.argument.split(",")])
        if self.kind == "ensemble":
            return np.eye(3)[AGGREGATION]
        return None

    def route(self, chat_prompts: list[str]) -> np.ndarray:
        if self.kind in ("expert", "base", "ensemble"):  # the controller plays no part
            return np.tile(self.fixed(), (len(chat_prompts), 1))
        return super().route(chat_prompts)

    def numerical_output(self, weights: np.ndarray) -> np.ndarray:
        if self.kind == "ensemble":
            return np.ones(len(weights), bool)
        if self.kind in ("expert", "base"):
            return np.full(len(weights), self.argument == "aggregation" and self.benchmark == "gift")
        if self.kind == "top1":
            return weights.argmax(1) == AGGREGATION
        return super().numerical_output(weights)

    def execution_weights(self, weights: np.ndarray) -> np.ndarray:
        if self.kind == "top1":
            return np.eye(3)[weights.argmax(1)]
        if self.kind == "fixed":
            return np.tile(self.fixed(), (len(weights), 1))
        return weights

    def _numerical(self, rows, weights, batch=16):
        if self.kind != "ensemble":
            return super()._numerical(rows, weights, batch)
        parts = self.ensemble(rows)
        if self.argument in ("reference", "router"):
            return parts["reference" if self.argument == "reference" else "pool"]
        if self.argument == "fnf":  # the pool where FnF has no model for the frequency and term
            return [np.where(np.isfinite(f), f, p) for f, p in zip(parts["fnf"], parts["pool"], strict=True)]
        count = dict(equal8=8, equal13=13)[self.argument]
        return [pool_quantiles(c[None, :count], np.full((1, count), 1 / count))[0] for c in parts["candidates"]]


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[2] not in ("gift", "cik", "tse"):
        sys.exit(__doc__)
    variant, benchmark, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    if variant == "joint":
        model = dict(TEEMOE_MODEL_CLASS="joint:JointTeeMoE", TEEMOE_MODEL_OPTIONS="{}")
    else:
        model = dict(TEEMOE_MODEL_CLASS="ablations:Ablation",
                     TEEMOE_MODEL_OPTIONS=json.dumps(dict(variant=variant, benchmark=benchmark)))
    path = os.pathsep.join([str(ROOT), str(ROOT / "paper"), os.environ.get("PYTHONPATH", "")])
    subprocess.run([sys.executable, "-m", f"teemoe.eval.{benchmark}", *rest],
                   env=dict(os.environ, **model, PYTHONPATH=path), check=True)


if __name__ == "__main__":
    main()
