"""The shared system prompt, the four user-message templates, and output parsing.

Every TeeMoE request (training or inference) is rendered through these
functions, so the text here must stay exactly as the released model saw it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

import numpy as np

from .evidence import local_period_text, polynomial_text, sketch

SYSTEM_PROMPT = (
    "First identify whether the raw request is time-series analysis or forecasting. If it is "
    "forecasting, choose structured aggregation versus native contextual forecasting from the "
    "supplied evidence, never from task identity or the mere presence of candidates.\n\n"
    "TIME-SERIES ANALYSIS: Derive the answer from the raw series and relevant supplied "
    "measurements. Use only diagnostics that bear on the requested property; do not let "
    "irrelevant scale, offset, amplitude, visual complexity, terminology, or incidental "
    "attributes override the numerical evidence. When options are present, decide which option "
    "text the evidence supports before mapping it to its label.\n\n"
    "STRUCTURED AGGREGATION: Compare supplied candidate forecast distributions and any protected "
    "aggregate. Make only calibrated, numerically supported changes through the structured "
    "forecasting output.\n\n"
    "NATIVE CONTEXTUAL FORECASTING: Generate one coherent plausible sample from the conditional "
    "future. Start from a continuation that preserves the target history's level, dynamics, "
    "dependence, seasonality, and uncertainty, and change it only where supplied context provides "
    "supported future evidence. Enforce exact values, equations, and bounds, apply interventions "
    "with their supported timing and magnitude, and ignore context that is merely descriptive or "
    "irrelevant.\n\n"
    "Return only the requested output. For a multiple-choice analysis, return exactly one line in "
    "the form `LETTER) exact option text`; do not explain or repeat the options. For a native "
    "forecast, return only (timestamp, value) pairs inside <forecast> and </forecast>."
)

CHOICE_SCHEMA = ("Return exactly one line in the form `LETTER) exact option text`; "
                 "do not explain or repeat the options.")
FREEFORM_SCHEMA = "Answer the analytical question directly."
FORECAST_HEADER = "I have a time series forecasting task for you.\n\n"
ANALYSIS_HEADER = "Analyze the supplied time-series evidence.\n\n"
MOLDS = ("context_aided_forecast", "non_context_forecast", "abcd_analysis", "freeform_analysis")

# A forecast value as the constrained decoder and the parser accept it.
NUMBER = r"[-+]?(?:(?:\d{1,18}(?:\.\d{1,12})?)|(?:\.\d{1,12}))(?:[eE][-+]?\d{1,2})?"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_FORECAST_HISTORY = 168


def render_user(row: dict) -> str:
    """Render one request dictionary into the user message of its template."""
    mold = row["mold"]
    if mold not in MOLDS:
        raise ValueError(f"unknown template: {mold!r}")
    evidence = str(row["evidence"]).strip()
    if not evidence:
        raise ValueError("evidence cannot be empty")
    if mold in {"abcd_analysis", "freeform_analysis"}:
        definitions = str(row.get("concept_definitions", ""))
        clarification = str(row.get("question_clarification", ""))
        if definitions:
            evidence += "\nSupplied concept definitions:\n" + definitions
        if clarification:
            evidence += "\nSupplied question clarification:\n" + clarification
    if mold == "context_aided_forecast" and {"context", "history_evidence"} <= row.keys():
        return (FORECAST_HEADER + "Context:\n<context>\n" + str(row["context"]).strip()
                + "\n</context>\n\n" + _history_block(row))
    if mold == "non_context_forecast" and "history_evidence" in row:
        return FORECAST_HEADER + _history_block(row)
    if mold in {"context_aided_forecast", "non_context_forecast"}:
        return ("Forecast the requested future values from the supplied evidence.\n\n"
                f"Evidence:\n{evidence}\n\nPrediction points:\n"
                + str(row["prediction_points"]).strip())
    question = str(row["question"]).strip()
    if mold == "abcd_analysis":
        schema = str(row.get("output_schema", "Return exactly one line as LETTER) exact option text.")).strip()
        return (ANALYSIS_HEADER + f"Evidence:\n{evidence}\n\nQuestion:\n{question}\n\n"
                f"Options:\n{str(row['options']).strip()}\n\nRequested output:\n{schema}")
    schema = str(row.get("output_schema", FREEFORM_SCHEMA)).strip()
    return (ANALYSIS_HEADER + f"Evidence:\n{evidence}\n\nQuestion:\n{question}\n\n"
            f"Requested output:\n{schema}")


def _history_block(row: dict) -> str:
    history, points = str(row["history_evidence"]).strip(), str(row["prediction_points"]).strip()
    if not history or not points:
        raise ValueError("forecast history and prediction points cannot be empty")
    return ("Historical time series in (timestamp, value) format:\n<history>\n" + history
            + "\n</history>\n\nFuture timestamps:\n" + points)


def format_value(value: float) -> str:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("forecast history contains a nonfinite value")
    return f"{value:.6g}" if value < 10**6 else f"{value:.0f}"


def calendar_timestamps(forecast_start, frequency: str, history_length: int,
                        horizon: int) -> tuple[list[str], list[str]]:
    """Past and future timestamps for a regular series whose forecast begins at ``forecast_start``."""
    import pandas as pd

    first = pd.Period(forecast_start, freq=frequency)
    stamp = lambda period: period.start_time.strftime(TIMESTAMP_FORMAT)
    past = pd.period_range(end=first - 1, periods=history_length, freq=frequency)
    future = pd.period_range(first, periods=horizon, freq=frequency)
    return [stamp(p) for p in past], [stamp(p) for p in future]


def history_timestamps(start, frequency: str, history_length: int,
                       horizon: int) -> tuple[list[str], list[str]]:
    """Past and future timestamps for a regular series whose first observation is at ``start``."""
    import pandas as pd

    first = pd.Period(start, freq=frequency) + history_length
    return calendar_timestamps(first.start_time, frequency, history_length, horizon)


def forecast_message(history: Sequence[float], past_timestamps: Sequence[str],
                     future_timestamps: Sequence[str], context: str = "",
                     max_history: int = MAX_FORECAST_HISTORY) -> str:
    """The forecasting user message: the last ``max_history`` observations and the future times."""
    values, stamps = list(history)[-max_history:], list(past_timestamps)[-max_history:]
    if len(values) != len(stamps) or not values or not future_timestamps:
        raise ValueError("history, timestamps and horizon must be nonempty and aligned")
    evidence = "\n".join(f"({t}, {format_value(v)})" for t, v in zip(stamps, values, strict=True))
    row = dict(mold="context_aided_forecast" if context.strip() else "non_context_forecast",
               evidence=evidence, history_evidence=evidence,
               prediction_points="\n".join(map(str, future_timestamps)))
    if context.strip():
        row["context"] = context.strip()
    return render_user(row)


def analysis_evidence(series: Sequence[Sequence[float]]) -> str:
    """Raw values plus the fixed, question-independent numerical measurements."""
    arrays = [np.asarray(values, dtype=np.float64) for values in series]
    if not 1 <= len(arrays) <= 2:
        raise ValueError("analysis requests take one or two series")
    sample = {"ts": arrays[0]} if len(arrays) == 1 else {"ts1": arrays[0], "ts2": arrays[1]}
    parts = []
    for index, values in enumerate(arrays, 1):
        name = "Time series" if len(arrays) == 1 else f"Time series {index}"
        parts.append(name + " values (successive observations separated by spaces):\n"
                     + " ".join(format(float(x), ".4g") for x in values))
    parts.append("Computed measurements:\n" + sketch(sample))
    parts.extend((polynomial_text(sample), local_period_text(sample)))
    return "\n".join(parts)


def analysis_message(series: Sequence[Sequence[float]], question: str,
                     options: Sequence[str] | None = None, *, concept_definitions: str = "",
                     clarification: str = "") -> str:
    """The analysis user message; multiple-choice when ``options`` are given."""
    row = dict(evidence=analysis_evidence(series), question=question,
               concept_definitions=concept_definitions, question_clarification=clarification)
    if options:
        row.update(mold="abcd_analysis", output_schema=CHOICE_SCHEMA,
                   options="\n".join(f"{chr(65 + i)}) {text}" for i, text in enumerate(options)))
    else:
        row.update(mold="freeform_analysis", output_schema=FREEFORM_SCHEMA)
    return render_user(row)


def chat_prompt(tokenizer, user: str, system: str = SYSTEM_PROMPT) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def routing_view(prompt: str) -> str:
    """The task-text view used for routing: the request without its signal values."""
    marker = "<|im_start|>user\n"
    before, _, rest = prompt.partition(marker)
    user, end, suffix = rest.partition("<|im_end|>")
    if not end:
        raise ValueError("routing expects one chat-formatted user message")
    if user.startswith(FORECAST_HEADER):
        head, found, tail = user.partition("Historical time series in (timestamp, value) format:\n<history>\n")
        if not found or "\n</history>\n\nFuture timestamps:\n" not in tail:
            raise ValueError("unknown forecasting request layout")
        user = head.rstrip()
    elif user.startswith(ANALYSIS_HEADER + "Evidence:\n"):
        evidence, found, question = user.partition("\n\nQuestion:\n")
        if not found:
            raise ValueError("analysis request lacks a question")
        match = re.search(r"(?:^|\n)(Supplied concept definitions:|Supplied question clarification:)\n",
                          evidence)
        context = [evidence[match.start():].strip()] if match else []
        user = "\n\n".join([ANALYSIS_HEADER.strip(), *context, "Question:\n" + question])
    else:
        raise ValueError("unknown request layout")
    return before + marker + user + end + suffix


def forecast_regex(timestamps: Sequence[str]) -> str:
    rows = "".join(rf"\(\s*{re.escape(str(t))}\s*,\s*{NUMBER}\)\n" for t in timestamps)
    return rf"<forecast>\n{rows}<\/forecast>"


def forecast_max_tokens(horizon: int) -> int:
    return max(512, 64 + 40 * horizon)


def parse_forecast(text: str, timestamps: Sequence[str]) -> list[float]:
    blocks = re.findall(r"<forecast>(.*?)</forecast>", text, re.I | re.S)
    if not blocks:
        raise ValueError("missing <forecast> block")
    pairs = re.findall(rf"\(\s*([^,\r\n]+?)\s*,\s*({NUMBER})\s*\)", blocks[-1], re.S)
    found = {stamp.strip().strip("'\""): float(value) for stamp, value in pairs}
    if missing := [t for t in timestamps if str(t) not in found]:
        raise ValueError(f"forecast lacks timestamps {missing[:3]}")
    values = [found[str(t)] for t in timestamps]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("forecast contains nonfinite values")
    return values


def parse_choice(text: str, options: Sequence[str]) -> int | None:
    """Index of the chosen option in a ``LETTER) option text`` answer, if one is identifiable."""
    for index, option in enumerate(options):
        if f"{chr(65 + index)}) {option}".casefold() in text.casefold():
            return index
    return None
