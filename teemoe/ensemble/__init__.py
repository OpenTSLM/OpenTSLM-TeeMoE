"""The numerical ensemble that supplies the aggregation expert's inputs.

For each request: 8 core forecasts are pooled with XGBoost weights, the pool is
blended with the released Toto-FnF ensemble, and the result (the *reference*
forecast) is refined by the aggregation expert together with 13 candidates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .fnf import FNF_MEMBERS, TotoFnF, blend, canonical_frequency, ensure_fnf
from .forecasters import CORE_MODELS, EXTRA_CANDIDATES
from .router import Router, pool_quantiles, router_features

CORE_ORDER = tuple(CORE_MODELS)  # router and editor order
CANDIDATE_ORDER = CORE_ORDER + EXTRA_CANDIDATES


def candidates_for(index: int, core: dict[str, list], fnf: dict[str, list]) -> np.ndarray:
    """[13, horizon, 9] sorted candidate deciles for request ``index`` (NaN where a model failed)."""
    stacked = np.stack([core[name][index] for name in CORE_ORDER]
                       + [fnf[name][index] for name in EXTRA_CANDIDATES]).astype(np.float32)
    return np.sort(stacked, axis=-1)


def valid(forecasts: np.ndarray) -> np.ndarray:
    """Which of the stacked forecasts [..., horizon, 9] are finite and of sane magnitude."""
    return np.isfinite(forecasts).all(axis=(-2, -1)) & (np.abs(forecasts) < 1e30).all(axis=(-2, -1))


class NumericalEnsemble:
    def __init__(self, router: str | Path, fnf_mass: float, fnf_root: str | Path | None = None,
                 device: str = "cpu"):
        self.router = Router(router, device=device)
        self.fnf_mass = float(fnf_mass)
        self.fnf = TotoFnF(fnf_root or ensure_fnf())

    def reference(self, requests: list[dict], core: dict[str, list], fnf: dict[str, list]):
        """Candidates [13, h, 9] and reference quantiles [h, 9] for every request."""
        parts = self.components(requests, core, fnf)
        return parts["candidates"], parts["reference"]

    def components(self, requests: list[dict], core: dict[str, list], fnf: dict[str, list]) -> dict[str, list]:
        """Per request: the 13 ``candidates``, the router ``pool`` of the core 8, the ``fnf``
        forecast (a request's ready ``fnf_forecast`` if it has one; NaN where unsupported) and
        their blend, the ``reference``."""
        candidates = [candidates_for(i, core, fnf) for i in range(len(requests))]
        for cand in candidates:  # a failed core model borrows the mean of the others for the features
            ok = valid(cand[:8])
            if not ok.any():
                raise ValueError("every core forecaster failed on a request")
            cand[:8][~ok] = cand[:8][ok].mean(axis=0)
        features = np.stack([router_features(r["history"], c[:8], frequency=canonical_frequency(r["frequency"]),
                                             term=r.get("term", "short"))
                             for r, c in zip(requests, candidates, strict=True)])
        weights = self.router.allocations(features)
        parts = dict(candidates=candidates, pool=[], fnf=[], reference=[])
        for i, (request, cand) in enumerate(zip(requests, candidates, strict=True)):
            left = pool_quantiles(cand[None, :8], weights[i:i + 1])[0]
            cand[8:][~valid(cand[8:])] = left  # a failed extra candidate is replaced by the pool
            frequency, term = canonical_frequency(request["frequency"]), request.get("term", "short")
            right = np.full_like(left, np.nan)
            if request.get("fnf_forecast") is not None:
                right = np.asarray(request["fnf_forecast"], np.float32)
            elif self.fnf.supports(frequency, term):
                frame = self.fnf.features(request["history"], frequency=frequency, horizon=int(request["horizon"]),
                                          domain=request.get("domain"))
                members = {m: np.asarray(fnf[m][i], np.float32).T[None] for m in FNF_MEMBERS}
                right = self.fnf.forecast(frame, members, frequency=frequency, term=term)[0].T
            parts["pool"].append(left)
            parts["fnf"].append(right)
            parts["reference"].append(blend(left, right, self.fnf_mass))
        return parts
