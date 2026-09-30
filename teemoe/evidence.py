"""Question-independent measurements appended to analysis requests.

These summaries are computed only from the observed values; they never read the
question, the options or the answer.
"""

from __future__ import annotations

import numpy as np


def _number(value: float) -> str:
    if not np.isfinite(value):
        return "0"
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-4):
        return f"{value:.4g}"
    return f"{value:.5f}".rstrip("0").rstrip(".")


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or len(b) != len(a) or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _slope(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    t = np.arange(len(x), dtype=np.float64)
    return float(np.dot(t - t.mean(), x - x.mean()) / max(np.dot(t - t.mean(), t - t.mean()), 1e-12))


def _acf(x: np.ndarray, lag: int) -> float:
    return _corr(x[:-lag], x[lag:]) if lag < len(x) else 0.0


def _period(x: np.ndarray) -> tuple[float, float]:
    n = len(x)
    if n < 8:
        return 0.0, 0.0
    t = np.arange(n, dtype=np.float64)
    detrended = x - (x.mean() + _slope(x) * (t - t.mean()))
    power = np.abs(np.fft.rfft(detrended)) ** 2
    power[0] = 0
    if len(power) <= 1 or power.sum() <= 1e-12:
        return 0.0, 0.0
    index = int(np.argmax(power[1:]) + 1)
    return float(n / index), float(power[index] / power.sum())


def _windows(x: np.ndarray) -> str:
    pieces = []
    for index, values in enumerate(np.array_split(x, 8), 1):
        pieces.append(
            f"w{index}(mean={_number(float(values.mean()))},sd={_number(float(values.std()))},"
            f"slope={_number(_slope(values))})"
        )
    return " ".join(pieces)


def _single(name: str, x: np.ndarray) -> str:
    diff = np.diff(x)
    q10, med, q90 = np.quantile(x, [0.1, 0.5, 0.9])
    period, power = _period(x)
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    robust_scale = 1.4826 * mad if mad > 1e-12 else max(float(np.std(x)), 1e-12)
    anomaly = int(np.argmax(np.abs(x - median)))
    lags = [lag for lag in (1, 2, 4, 8, 16, 32) if lag < len(x)]
    acfs = ",".join(f"{lag}:{_number(_acf(x, lag))}" for lag in lags)
    return (
        f"{name}: n={len(x)} mean={_number(float(x.mean()))} sd={_number(float(x.std()))} "
        f"min={_number(float(x.min()))} q10={_number(float(q10))} median={_number(float(med))} "
        f"q90={_number(float(q90))} max={_number(float(x.max()))} slope={_number(_slope(x))} "
        f"diff_sd={_number(float(diff.std()) if len(diff) else 0)} "
        f"diff_acf1={_number(_acf(diff, 1) if len(diff) > 1 else 0)} raw_acf[{acfs}] "
        f"fft_period={_number(period)} fft_power_fraction={_number(power)} "
        f"largest_robust_z={_number(float(abs(x[anomaly] - median) / robust_scale))} "
        f"at_fraction={_number(anomaly / max(len(x) - 1, 1))}\n"
        f"{name}_eight_windows: {_windows(x)}"
    )


def _shifted_correlations(a: np.ndarray, b: np.ndarray) -> list[tuple[float, int]]:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    minimum = max(32, n // 3)
    maximum_lag = max(0, min(128, n - minimum))
    values = []
    for lag in range(-maximum_lag, maximum_lag + 1):
        if lag >= 0:
            left, right = a[: n - lag], b[lag:]
        else:
            left, right = a[-lag:], b[: n + lag]
        values.append((_corr(left, right), lag))
    return sorted(values, key=lambda item: (-abs(item[0]), abs(item[1]), item[1]))[:3]


def _predictive_gain(source: np.ndarray, target: np.ndarray) -> tuple[float, int]:
    n = min(len(source), len(target))
    source, target = source[:n], target[:n]
    maximum_lag = min(32, n // 4)
    y = target[maximum_lag:]
    own = target[maximum_lag - 1 : n - 1]
    base = np.column_stack([np.ones(len(y)), own])
    beta0 = np.linalg.lstsq(base, y, rcond=None)[0]
    rss0 = float(np.sum((y - base @ beta0) ** 2))
    best = (0.0, 0)
    for lag in range(1, maximum_lag + 1):
        other = source[maximum_lag - lag : n - lag]
        full = np.column_stack([np.ones(len(y)), own, other])
        beta1 = np.linalg.lstsq(full, y, rcond=None)[0]
        rss1 = float(np.sum((y - full @ beta1) ** 2))
        gain = max(0.0, (rss0 - rss1) / max(rss0, 1e-12))
        if gain > best[0]:
            best = (gain, lag)
    return best


def _pair(a: np.ndarray, b: np.ndarray) -> str:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    shifted = _shifted_correlations(a, b)
    shifted_text = ", ".join(f"lag={lag}:r={_number(value)}" for value, lag in shifted)
    da, db = np.diff(a), np.diff(b)
    diff_shift = _shifted_correlations(da, db)[0] if len(da) >= 3 else (0.0, 0)
    gain12, lag12 = _predictive_gain(a, b)
    gain21, lag21 = _predictive_gain(b, a)
    design = np.column_stack([np.ones(n), a])
    prediction = design @ np.linalg.lstsq(design, b, rcond=None)[0]
    r2 = 1.0 - float(np.sum((b - prediction) ** 2)) / max(float(np.sum((b - b.mean()) ** 2)), 1e-12)
    return (
        "Pair convention: positive lag k means S1[t] aligns with S2[t+k] (S1 leads S2).\n"
        f"Pair: corr0={_number(_corr(a,b))}; top_shifted_corr=[{shifted_text}]; "
        f"best_diff_corr=lag={diff_shift[1]}:r={_number(diff_shift[0])}; "
        f"affine_S1_to_S2_R2={_number(r2)}; "
        f"incremental_prediction_S1_to_S2=gain={_number(gain12)},lag={lag12}; "
        f"incremental_prediction_S2_to_S1=gain={_number(gain21)},lag={lag21}."
    )


def streams(sample: dict) -> list[np.ndarray]:
    raw = [sample["ts"]] if "ts" in sample else [sample["ts1"], sample["ts2"]]
    arrays = [np.asarray(value, dtype=np.float64) for value in raw]
    if any(x.ndim != 1 or len(x) < 8 or not np.all(np.isfinite(x)) for x in arrays):
        raise ValueError("invalid presented time series")
    return arrays


def sketch(sample: dict) -> str:
    values = streams(sample)
    lines = [_single(f"S{index + 1}", value) for index, value in enumerate(values)]
    if len(values) == 2:
        lines.append(_pair(values[0], values[1]))
    return "\n".join(lines)

def polynomial_diagnostics(values):
    x = np.asarray(values, dtype=np.float64)
    scale = float(x.std())
    if scale < 1e-12:
        return None
    z = (x - x.mean()) / scale
    time = np.linspace(-1.0, 1.0, len(z))
    design = np.column_stack([np.ones(len(z)), time, time ** 2])
    linear = np.linalg.lstsq(design[:, :2], z, rcond=None)[0]
    quadratic = np.linalg.lstsq(design, z, rcond=None)[0]
    total = float(z @ z)
    return dict(
        linear_R2=1 - float(np.sum((z-design[:, :2] @ linear) ** 2)) / total,
        quadratic_R2=1 - float(np.sum((z-design @ quadratic) ** 2)) / total,
        quadratic_t2=float(quadratic[2]))


def polynomial_text(sample):
    lines = ['Trend fits: time rescaled to [-1,1], values standardized; '
             'R2 is variance explained; quadratic_t2 is the coefficient of time squared.']
    for index, values in enumerate(streams(sample), 1):
        result = polynomial_diagnostics(values)
        text = ('unavailable (constant series)' if result is None else
                ' '.join(f'{key}={value:.5g}' for key, value in result.items()))
        lines.append(f'S{index}_trend_fits: {text}')
    return '\n'.join(lines)


def local_period_text(sample):
    lines = [
        'Local spectrum: consecutive first and second halves, each linearly '
        'detrended. FFT period is in samples; power_fraction is the fraction '
        'of non-DC spectral power at that peak. Peaks are estimates, not '
        'proof of periodicity.'
    ]
    for index, values in enumerate(streams(sample), 1):
        for name, half in zip(('first_half', 'second_half'), np.array_split(values, 2)):
            period, fraction = _period(half)
            fields = ('unavailable (constant or insufficient signal)' if fraction == 0 else
                      f'fft_period={period:.5g} power_fraction={fraction:.5g}')
            lines.append(f'S{index}_{name}: n={len(half)} {fields}')
    return '\n'.join(lines)


def enrich_source_evidence(evidence, values):
    """Append the same fixed measurements to source-declared temporal channels."""
    if evidence.startswith('Sensor measurements (mg), not a sequence of nine times:'):
        return evidence
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite source observations')
    channels = ([(name, values[i::3]) for i, name in enumerate('XYZ')]
                if evidence.startswith('Accelerometer observations at successive times (20 Hz).')
                else [('S1', values)])
    sections = []
    for function in (polynomial_text, local_period_text):
        for i, (name, channel) in enumerate(channels):
            lines = function({'ts': channel}).splitlines()
            if i == 0:
                sections.append(lines[0])
            sections.extend(line.replace('S1_', name+'_', 1) for line in lines[1:])
    return evidence+'\n'+'\n'.join(sections)
