"""Training examples for the native (contextual) forecasting expert.

    python -m teemoe.data.native --output data/native/train.jsonl

  source                                                      examples
  TADiff-Synth captioned pairs (both members of each pair)      7,200
  NWS forecast discussions with ASOS hourly temperatures        2,800
  CAF-7M contextual scenarios over Chronos datasets             2,800
  Pierrot & Pinson bounded simulations with their bounds        3,600
  LOTSA windows with imposed future bounds                      3,600

Every example is a rendered request (``mold``, ``context``, ``history_evidence``,
``prediction_points``) and its ``target`` answer; histories show at most the
last 168 observations and futures at most 64 steps (TADiff and the bounded
simulations are shown whole). Downloads are cached under
``--sources`` and each source's examples under ``--output``'s directory.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
import math
import random
import re
import struct
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from ..prompts import format_value
from .windows import frequency_regime, log, repair, stable_seed, write_jsonl

MAX_HISTORY, MAX_HORIZON = 168, 64
DOWNLOADS = {
    "tadiff/train_ts.npy": ("https://drive.usercontent.google.com/download?id=16qVVExcys7UaRCZKeMxqLJBHqSv2WK7w"
                            "&export=download", "7250d579745177591b0dceb76b02dcbcf9b2832e9be04f9ba833c72bae5fd998"),
    "tadiff/train_caps.npy": ("https://drive.usercontent.google.com/download?id=178j6rlzi7cbdc9_aW1DAyPdazJfUntJu"
                              "&export=download", "9d3084a6ac39fc7e599103319c37f80db42d6fc7d7013c648140845fbf40986f"),
    "bounded/R_project.zip": ("https://ndownloader.figshare.com/files/46071161",
                              "9a42bcdf3622f79fe8a8d14e7f8fd5afe9a531a93c35dd04a8187d25fbec6b8f"),
}
# NWS forecast office -> ASOS station (Iowa Environmental Mesonet archives, 2022-2024)
WEATHER = dict(DMX="DSM", OUN="OKC", BOX="BOS", MFL="MIA", SEW="SEA", PSR="PHX", SLC="SLC", BOU="DEN",
               LWX="DCA", HGX="IAH", SGF="SGF", BUF="BUF")
AFD_URL = ("https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py?pil=AFD{office}&fmt=zip&limit=9999"
           "&sdate=2022-01-01&edate=2025-01-01")
ASOS_URL = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={station}&data=tmpf&year1=2022"
            "&month1=1&day1=1&year2=2025&month2=1&day2=1&tz=Etc%2FUTC&format=onlycomma&latlon=no&elev=no"
            "&missing=empty&trace=empty&direct=no&report_type=2")
CAF = dict(repo="ServiceNow/CAF_7M", revision="1ea13277164f06e161435c9e37f348328be61c03",
           series_repo="autogluon/chronos_datasets", series_revision="eeecad0b82a8c237e212ce6f8d1abecb513e2cec")
LOTSA = dict(repo="Salesforce/lotsa_data", revision="8191fd29eb5cf906ec55effca44d8059888b615d")
LOTSA_DATASETS = (
    "bdg-2_fox", "bdg-2_rat", "bitcoin_with_missing", "borealis", "buildings_900k", "bull", "cif_2016_12",
    "covid_mobility", "elecdemand", "favorita_sales", "favorita_transactions", "hog", "ideal",
    "kaggle_web_traffic_weekly", "lcl", "m1_monthly", "m1_yearly", "m5", "monash_m3_other", "monash_m3_quarterly",
    "monash_m3_yearly", "nn5_daily_with_missing", "nn5_weekly", "oikolab_weather", "pdb", "project_tycho",
    "rideshare_with_missing", "smart", "spain", "subseasonal_precip", "sunspot_with_missing", "taxi_30min",
    "tourism_monthly", "tourism_yearly", "traffic_weekly", "uber_tlc_hourly", "vehicle_trips_with_missing",
    "weather", "wiki-rolling_nips")
# LOTSA forecast horizons by sampling frequency
LOTSA_HORIZONS = dict(minute=(48,), hour=(24, 48), day=(6, 12, 24), week=(6, 12, 24), month=(12, 24, 48),
                      quarter=(6, 12), year=(6,))


# ------------------------------------------------------------------ rendering
def stamp(value) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S" if value.hour or value.minute or value.second else "%Y-%m-%d")


def example(row_id: str, history, future, past_stamps, future_stamps, context: str = "", *,
            history_format=format_value, source: str, cap: bool = True) -> dict:
    """One rendered native-forecasting example (with ``cap``: the last 168 observations and at
    most 64 future steps)."""
    history, past_stamps, future, future_stamps = map(list, (history, past_stamps, future, future_stamps))
    if cap:
        future, future_stamps = future[:MAX_HORIZON], future_stamps[:MAX_HORIZON]
        history, past_stamps = history[-MAX_HISTORY:], past_stamps[-MAX_HISTORY:]
    pairs = lambda stamps, values: [f"({t}, {history_format(v)})" for t, v in zip(stamps, values, strict=True)]
    evidence = "\n".join(pairs(past_stamps, history))
    row = dict(row_id=row_id, mold="context_aided_forecast" if context else "non_context_forecast",
               evidence=evidence, history_evidence=evidence, prediction_points="\n".join(map(str, future_stamps)),
               target="<forecast>\n" + "\n".join(pairs(future_stamps, future)) + "\n</forecast>", source=source)
    if context:
        row["context"] = row["evidence"] = context.strip()
    return row


def fetch(sources: Path, key: str) -> Path:
    """A pinned release file from ``DOWNLOADS``, downloaded once."""
    url, sha256 = DOWNLOADS[key]
    return download(url, sources / key, sha256)


def download(url: str, path: Path, sha256: str | None = None) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(url, timeout=300) as response, path.with_suffix(".part").open("wb") as stream:
            while chunk := response.read(1 << 20):
                stream.write(chunk)
        path.with_suffix(".part").rename(path)
    if sha256 and hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise ValueError(f"{path} differs from the pinned release")
    return path


# --------------------------------------------------------------------- TADiff
def tadiff(sources: Path, count: int = 7200, seed: int = 0) -> list[dict]:
    """Captioned synthetic pairs; both members of each sampled history pair are kept."""
    series = np.load(fetch(sources, "tadiff/train_ts.npy"))
    captions = np.load(fetch(sources, "tadiff/train_caps.npy"))
    pairs = defaultdict(list)
    for index, values in enumerate(series):
        pairs[hashlib.sha256(np.asarray(values[:128], "<f8").tobytes()).hexdigest()].append(index)
    chosen = random.Random(seed).sample(sorted(pairs), count // 2)
    steps = list(range(256))
    return [example(f"tadiff-{i:05d}", series[i][:128], series[i][128:], steps[:128], steps[128:],
                    str(captions[i][1]), history_format=lambda v: f"{float(v):.6g}", source="tadiff", cap=False)
            for key in chosen for i in pairs[key]]


# -------------------------------------------------------------------- bounded
def bounded(sources: Path, count: int = 3600, seed: int = 0) -> list[dict]:
    """Pierrot & Pinson's AR(1)-sinusoid simulations on (0, b_t), with b_t supplied as context."""
    # The values and bounds sit at fixed offsets of the pinned (XDR-serialized) R matrix.
    with zipfile.ZipFile(fetch(sources, "bounded/R_project.zip")) as zipped:
        raw = gzip.decompress(zipped.read("R_project/data/mytoydata_AR(1)_sinusoid.b--MC.RDS"))
    if struct.unpack_from(">ii", raw, 31) != (0x20E, 1200000):
        raise ValueError("unexpected simulation file layout")
    values = np.frombuffer(raw, ">f8", 1200000, 39).reshape((12000, 100), order="F").astype(np.float64)
    bounds = np.frombuffer(raw, ">f8", 12000, 9600086).astype(np.float64)
    history, width = 128, 128 + 32  # 128 observed steps, 32 forecast
    core = np.linspace(0, len(bounds) - width, 20, dtype=int).tolist()  # 56 disjoint windows per series
    gaps, cursor = [], 0
    for start in core + [len(bounds)]:
        gaps += range(cursor, min(start - width, len(bounds) - width) + 1, width)
        cursor = start + width
    starts = sorted(core + [gaps[i] for i in np.linspace(0, len(gaps) - 1, 36, dtype=int)])
    pool = [(s, start) for s in range(values.shape[1]) for start in starts]
    exact = lambda v: f"{float(v):.17g}"
    rows = []
    for s, start in random.Random(seed).sample(pool, count):
        cut, stop = start + history, start + width
        context = "X_t in (0, b_t)\nb_t (upper support; time index, value):\n" + "\n".join(
            f"({t + 1}, {exact(bounds[t])})" for t in range(start, stop))
        rows.append(example(f"bounded-{s + 1:03d}-{cut:05d}", values[start:cut, s], values[cut:stop, s],
                            range(start + 1, cut + 1), range(cut + 1, stop + 1), context,
                            history_format=exact, source="bounded", cap=False))
    return rows


# ---------------------------------------------------------------------- NWS
SECTION = re.compile(r"(?im)^\.(?:SYNOPSIS|DISCUSSION|NEAR TERM|SHORT TERM|LONG TERM)\b")
STOP_NAMES = (r"AVIATION|MARINE|FIRE WEATHER|HYDROLOGY|CLIMATE|PRELIMINARY POINT TEMPS/POPS|"
              r"[A-Z]{2,3} WATCHES/WARNINGS/ADVISORIES")
STOP = re.compile(r"(?im)^\s*\.{1,3}(?:" + STOP_NAMES + r")\b")
INLINE_STOP = re.compile(r"(?i)\.{1,3}(?:" + STOP_NAMES + r")\.{2,3}")


def discussion_text(raw: str) -> str | None:
    """The synopsis and forecast sections of an area forecast discussion (None if too short)."""
    text = raw.replace("\r", "\n").replace("\x00", " ")
    if (start := SECTION.search(text)) is None:
        return None
    text = text[start.start():]
    if stop := STOP.search(text):
        text = text[:stop.start()]
    text = re.sub(r"(?m)^Issued at .*?$", "", text)
    text = re.sub(r"\s+", " ", re.sub(r"(?m)^&&\s*$", "", text)).strip()
    if stop := INLINE_STOP.search(text):
        text = text[:stop.start()].strip()
    return text[:12000] if len(text.split()) >= 80 else None


def weather(sources: Path, count: int = 2800, seed: int = 0) -> list[dict]:
    """The daily discussion issued nearest 12 UTC, with 24 hours of temperatures before and after."""
    pools = {}
    for office, station in WEATHER.items():
        root = sources / "weather" / office
        afd = download(AFD_URL.format(office=office), root / f"AFD{office}_2022_2024.zip")
        asos = download(ASOS_URL.format(station=station), root / f"{station}_2022_2024.csv")
        reports = {}
        with zipfile.ZipFile(afd) as archive:
            for name in archive.namelist():
                if not (match := re.search(r"_(\d{12})\.txt$", name)):
                    continue
                issued = datetime.strptime(match.group(1), "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
                text = discussion_text(archive.read(name).decode("utf-8", errors="replace"))
                old = reports.get(issued.date())
                if text and (old is None or abs(issued.hour + issued.minute / 60 - 12)
                             < abs(old[0].hour + old[0].minute / 60 - 12)):
                    reports[issued.date()] = issued, text
        nearest = {}
        for row in csv.DictReader(io.StringIO(asos.read_bytes().decode("utf-8", errors="replace"))):
            if not row.get("tmpf"):
                continue
            time = datetime.strptime(row["valid"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            hour = (time + timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)
            value = float(row["tmpf"])
            if math.isfinite(value) and -100 <= value <= 140 and (
                    hour not in nearest or abs((time - hour).total_seconds()) < nearest[hour][0]):
                nearest[hour] = abs((time - hour).total_seconds()), value
        hourly = {hour: value for hour, (_, value) in nearest.items()}
        pools[office] = []
        for issued, text in reports.values():
            anchor = issued.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            past = [anchor - timedelta(hours=i) for i in range(24, 0, -1)]
            future = [anchor + timedelta(hours=i) for i in range(24)]
            if not all(t in hourly for t in past + future):
                continue
            context = "\n".join(["Measured variable: air temperature", "Units: degrees Fahrenheit",
                                 f"Observation station: {station}", f"Weather forecast office: {office}",
                                 f"Report issued at: {issued.isoformat()}"]) + "\n\n" + text
            pools[office].append(example(
                f"nws-{office}-{issued:%Y%m%d%H%M}", [hourly[t] for t in past], [hourly[t] for t in future],
                [t.isoformat() for t in past], [t.isoformat() for t in future], context, source="nws"))
        random.Random(stable_seed(seed, office)).shuffle(pools[office])
        log(source="nws", office=office, windows=len(pools[office]))
    offices = sorted(pools)
    ordered = [pools[o][i] for i in range(max(map(len, pools.values()))) for o in offices if i < len(pools[o])]
    return ordered[:count]


# ---------------------------------------------------------------------- CAF
def caf_selection(count: int = 2800, seed: int = 0) -> list[tuple[str, int]]:
    """Select only from the released model's cleaned CAF rows (zero-based Parquet positions)."""
    selected = json.loads(Path(__file__).with_name("caf_rows.json").read_text())
    rows = [(name, index) for name, indices in selected.items() for index in indices]
    if not 0 <= count <= len(rows):
        raise ValueError(f"CAF count must be between 0 and {len(rows)}")
    random.Random(seed).shuffle(rows)
    return rows[:count]


def caf(sources: Path, count: int = 2800, seed: int = 0) -> list[dict]:
    """Reconstruct the cleaned CAF-7M selection from its public metadata and source series."""
    import pyarrow.parquet as pq
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    ordered = caf_selection(count, seed)
    if not ordered:
        return []
    filenames = sorted({name for name, _ in ordered})
    root = Path(snapshot_download(CAF["repo"], repo_type="dataset", revision=CAF["revision"],
                                  allow_patterns=[f"data/{name}" for name in filenames], local_dir=sources / "caf"))
    meta, wanted = {}, defaultdict(list)
    for name in filenames:
        rows = sorted(r for f, r in ordered if f == name)
        table = pq.read_table(root / "data" / name).take(rows).to_pylist()
        meta.update({(name, r): m for r, m in zip(rows, table, strict=True)})
    for key in ordered:
        wanted[(meta[key]["dataset_name"], int(meta[key]["series_idx"]))].append(key)
    examples = {}
    for name in sorted({d for d, _ in wanted}):
        indices = {i for d, i in wanted if d == name}
        stream = load_dataset(CAF["series_repo"], name, split="train", revision=CAF["series_revision"],
                              streaming=True)
        for index, series in enumerate(stream):
            for key in wanted.get((name, index), ()):
                row = caf_example(f"caf-{key[0].split('-')[1]}-{key[1]}", meta[key], series)
                if row is None:
                    raise ValueError(f"Incomplete CAF source window: {key}")
                examples[key] = row
            indices.discard(index)
            if not indices:
                break
        log(source="caf", dataset=name, examples=len(examples))
    if len(examples) != count:
        raise ValueError(f"Only reconstructed {len(examples)} of {count} selected CAF examples")
    return [examples[key] for key in ordered]


def caf_example(row_id: str, meta: dict, series: dict) -> dict | None:
    """The example for one CAF-7M metadata row and its Chronos-dataset series (None if incomplete)."""
    past, future = _list(meta["past_timestamp"]), _list(meta["future_timestamp"])
    values = np.asarray(series[meta["target_column"]], np.float64)
    start, cut = int(meta["start_idx"]), int(meta["start_idx"]) + len(past)
    window = values[start:cut + len(future)]
    if len(window) != len(past) + len(future) or not np.isfinite(window).all():
        return None
    context = "\n".join(line for line in str(meta["context"]).strip().splitlines()
                        if not line.startswith("Temporal alignment known when forecasting:"))
    return example(row_id, window[:len(past)], window[len(past):], past, future, context.strip(), source="caf")


def _list(value) -> list[str]:
    return list(value) if not isinstance(value, str) else list(ast.literal_eval(value))


# --------------------------------------------------------------------- LOTSA
def bounds_context(row: dict, seed: int) -> dict:
    """Impose a random lower and/or upper limit (history quantiles) on the future and state it."""
    rng = random.Random(hashlib.sha256(f"{seed}\0{row['row_id']}".encode()).hexdigest() + ":limits")
    history = np.asarray([float(v) for v in re.findall(r", ([^,()\s]+)\)$", row["history_evidence"], re.M)])
    mode = rng.choice(("lower", "upper", "both"))
    levels = sorted((format(float(np.quantile(history, rng.uniform(0.1, 0.9))), ".6g")
                     for _ in range(2 if mode == "both" else 1)), key=float)
    lower = levels[0] if mode in ("lower", "both") else None
    upper = levels[-1] if mode in ("upper", "both") else None
    limits = ([f"a minimum of {lower}"] if lower else []) + ([f"a maximum of {upper}"] if upper else [])
    context = ("Starting at the first requested future timestamp, the reported series will have "
               + " and ".join(limits) + ". Values outside these limits are replaced by the nearest limit; "
               "values within them are unchanged. These limits apply throughout the requested future period, "
               "not to the historical observations.")
    target = []
    for stamp_, value in re.findall(r"^\(([^\n]+),\s*([^,()\s]+)\)$", row["target"], re.M):
        clipped = lower if lower is not None and float(value) < float(lower) else \
            upper if upper is not None and float(value) > float(upper) else value
        target.append(f"({stamp_}, {clipped})")
    return dict(row, mold="context_aided_forecast", context=context, evidence=context,
                target="<forecast>\n" + "\n".join(target) + "\n</forecast>")


def lotsa_window(row_id: str, target, frequency: str, start, cutoff: int, length: int, horizon: int) -> dict | None:
    """A plain window ``length`` steps before and ``horizon`` steps after ``cutoff``, dated from the
    series start."""
    import pandas as pd

    history, future = repair(target[cutoff - length:cutoff]), repair(target[cutoff:cutoff + horizon])
    if history is None or future is None or len(future) != horizon:
        return None
    offset = pd.tseries.frequencies.to_offset(frequency)
    first = pd.date_range(start=str(start or "2000-01-03"), periods=1, freq=offset)[0]
    stamps = [stamp(t) for t in pd.date_range(start=first + (cutoff - length) * offset,
                                              periods=length + horizon, freq=offset)]
    return example(row_id, history, future, stamps[:length], stamps[length:], source="lotsa")


def lotsa(count: int = 3600, windows_per_series: int = 4, max_series: int = 2000, seed: int = 0) -> list[dict]:
    """Windows of 2-16 horizons of history from LOTSA series, with imposed bounds as context."""
    from datasets import load_dataset

    per_dataset = math.ceil(count / len(LOTSA_DATASETS))
    pools = {}
    for name in LOTSA_DATASETS:
        rng, rows = random.Random(stable_seed(seed, name)), []
        stream = load_dataset(LOTSA["repo"], name, split="train", revision=LOTSA["revision"], streaming=True,
                              trust_remote_code=True)
        for index, source in enumerate(stream):
            if index >= max_series or len(rows) >= 4 * per_dataset:
                break
            frequency = str(source["freq"])
            target = np.asarray(source["target"], np.float32)
            try:
                horizons = LOTSA_HORIZONS[frequency_regime(frequency)]
            except (KeyError, ValueError):
                continue
            if target.ndim != 1:
                continue
            for _ in range(windows_per_series):
                horizon = rng.choice(horizons)
                if len(target) < horizon + 12:
                    break
                cutoff = rng.randint(12, len(target) - horizon)
                length = min(cutoff, rng.choice((2, 4, 8, 16)) * horizon)
                row = lotsa_window(f"lotsa-{name}-{index}-{cutoff}-{horizon}", target, frequency,
                                   source.get("start"), cutoff, length, horizon)
                if row is not None:
                    rows.append(row)
        rng.shuffle(rows)
        pools[name] = [bounds_context(row, seed) for row in rows[:per_dataset]]
        log(source="lotsa", dataset=name, windows=len(pools[name]))
    names = sorted(pools)
    ordered = [pools[n][i] for i in range(per_dataset) for n in names if i < len(pools[n])]
    return ordered[:count]


# ------------------------------------------------------------------------ CLI
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", default="data/sources/native", help="download cache")
    parser.add_argument("--output", default="data/native/train.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    sources, output = Path(args.sources), Path(args.output)
    parts = []
    for name, build in (("tadiff", tadiff), ("nws", weather), ("caf", caf), ("bounded", bounded),
                        ("lotsa", lambda s, seed: lotsa(seed=seed))):
        path = output.parent / "sources" / ("caf_selected.jsonl" if name == "caf" else f"{name}.jsonl")
        if not path.exists():
            write_jsonl(path, build(sources, seed=args.seed))
        parts += [json.loads(line) for line in path.open()]
        log(source=name, examples=sum(1 for _ in path.open()))
    random.Random(args.seed).shuffle(parts)
    print(json.dumps(dict(output=str(output), examples=write_jsonl(output, parts))))


if __name__ == "__main__":
    main()
