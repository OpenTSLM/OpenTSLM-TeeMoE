"""Training windows for the numerical ensemble and the aggregation expert.

    python -m teemoe.data.aggregation --gift-root third_party/gift-eval --gift-data data/gift-eval \
        --output data/aggregation

Writes one JSONL file per source block under ``--output/blocks``:

  block          source                                          windows
  gift_sampled   GIFT-Eval training split, 2,835 per cell        ~275,000
  gift_rolling   GIFT-Eval training split, rolling origins       ~31,000
  boom           Datadog/BOOM observability metrics              100,000
  rmisc          nine RMISC collections                           49,000
  lotsa          LOTSA, seven domains                            100,000
  diverse        UTSD-1G and RMISC retail (Dominick, Rossmann)    26,920

Only the GIFT-Eval *training* split is read (everything before the test
windows). Every block is sampled with a fixed seed; completed blocks are reused.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .windows import (balanced, channels, clean, gift_dataset, gift_eval, gift_metadata,
                      history_digest, log, repair, sample_windows, stable_seed, timestamp, write_jsonl)

BLOCKS = ("gift_sampled", "boom", "rmisc", "lotsa", "diverse", "gift_rolling")
DOMAINS = dict(econ_fin="Econ/Fin", energy="Energy", healthcare="Healthcare", nature="Nature",
               sales="Sales", transport="Transport", web_cloudops="Web/CloudOps")
MAX_HISTORY = 8192

BOOM = dict(repo="Datadog/BOOM", revision="69325b544c45ff0d6c43c7a99c49a6601a01725b")
RMISC = dict(repo="nju-zhangsq/RMISC", revision="29c8dbd53bcf704c9934a5cf18e03b7105fec77b")
# root: (frequency, domain, target columns: "targets" = the metadata's targets, "numeric" = all, or a regex)
RMISC_ROOTS = {
    "GasSensorTemperature": ("S", "energy", "targets"),
    "OccupancyDetection": ("T", "energy", "targets"),
    "MotorTemperature": ("S", "energy", "targets"),
    "WeeklyFuelPricesItaly": ("W", "econ_fin", "numeric"),
    "USAirPollution": ("D", "nature", r"^(?!.*Max Hour).*$"),
    "OPSD": ("H", "energy", r"(_load_actual_|_solar_generation_actual|_wind.*generation_actual)"),
    "HungarianChickenpoxCases": ("W", "healthcare", "targets"),
    "Pvdaq": ("15T", "energy", "numeric"),
    "MetroTraffic": ("H", "transport", "targets"),
}
RETAIL_ROOTS = {"Dominick": ("W", "econ_fin", "numeric"), "Rossmann_1W": ("W", "econ_fin", "targets")}
LOTSA = dict(repo="Salesforce/lotsa_data", revision="8191fd29eb5cf906ec55effca44d8059888b615d")
# domain -> source families -> LOTSA datasets (GIFT-Eval test datasets are excluded).
LOTSA_DOMAINS = {
    "econ_fin": {"m1": ["m1_monthly", "m1_quarterly", "m1_yearly"],
                 "m3": ["monash_m3_monthly", "monash_m3_other", "monash_m3_quarterly", "monash_m3_yearly"],
                 "cif_2016": ["cif_2016_6", "cif_2016_12"], "bitcoin_with_missing": ["bitcoin_with_missing"],
                 "fred_md": ["fred_md"], "sunspot_with_missing": ["sunspot_with_missing"]},
    "energy": {"elecdemand": ["elecdemand"], "elf": ["elf"], "lcl": ["lcl"],
               "bdg_2": ["bdg-2_bear", "bdg-2_fox", "bdg-2_panther", "bdg-2_rat"],
               "buildings_900k": ["buildings_900k"], "spain": ["spain"],
               "australian_electricity_demand": ["australian_electricity_demand"],
               "grid_competitions": ["gfc12_load", "gfc14_load", "gfc17_load"],
               "london_smart_meters_with_missing": ["london_smart_meters_with_missing"],
               "residential_load_power": ["residential_load_power"],
               "residential_generation": ["residential_pv_power", "solar_power", "wind_power"],
               "wind_farms_with_missing": ["wind_farms_with_missing"]},
    "healthcare": {"project_tycho": ["project_tycho"], "covid_mobility": ["covid_mobility"]},
    "nature": {"oikolab_weather": ["oikolab_weather"], "weather": ["weather"],
               "beijing_air_quality": ["beijing_air_quality"], "china_air_quality": ["china_air_quality"],
               "era5_2018": ["era5_2018"]},
    "sales": {"m5": ["m5"], "nn5_cash_demand": ["nn5_daily_with_missing", "nn5_weekly"]},
    "transport": {"pems_california": ["PEMS03", "PEMS04", "PEMS07", "PEMS08", "PEMS_BAY"],
                  "LOS_LOOP": ["LOS_LOOP"], "metro": ["HZMETRO", "SHMETRO", "BEIJING_SUBWAY_30MIN"],
                  "rideshare_with_missing": ["rideshare_with_missing"],
                  "tlc": ["taxi_30min", "uber_tlc_daily", "uber_tlc_hourly"],
                  "vehicle_trips_with_missing": ["vehicle_trips_with_missing"],
                  "pedestrian_counts": ["pedestrian_counts"], "traffic_hourly": ["traffic_hourly"],
                  "traffic_weekly": ["traffic_weekly"]},
    "web_cloudops": {"azure_vm_traces_2017": ["azure_vm_traces_2017"], "google_borg": ["borg_cluster_data_2011"],
                     "web_traffic": ["kaggle_web_traffic_weekly", "extended_web_traffic_with_missing",
                                     "wiki-rolling_nips"], "godaddy": ["godaddy"]},
}
UTSD = dict(repo="thuml/UTSD", revision="7326ff5f4578da73d843fd675d760c6c6054017f")
# UTSD-1G family: (frequency, domain); sub-second recordings are treated as secondly.
UTSD_FAMILIES = {
    "Health_IEEEPPG": ("S", "healthcare"), "Health_SelfRegulationSCP1": ("S", "healthcare"),
    "Health_SelfRegulationSCP2": ("S", "healthcare"), "Health_TDBrain_csv": ("S", "healthcare"),
    "Health_AtrialFibrillation": ("S", "healthcare"), "Nature_Worms": ("S", "nature"),
    "IoT_baian": ("S", "web_cloudops"), "Environment_BenzeneConcentration": ("H", "nature"),
    "Environment_AustraliaRainfall": ("H", "nature"),
}


def group_of(*parts) -> str:
    return "/".join(map(str, parts))


# ------------------------------------------------------------------ GIFT-Eval
def gift_sampled(root, data, *, rows_per_cell=3150, validation_fraction=0.1, max_history=MAX_HISTORY,
                 seed=20260802):
    """Random cutoffs in each cell's training split, balanced to the same count per cell.
    One tenth of each cell's distinct cutoffs is held out and never used."""
    Dataset, cells, properties = gift_eval(root, data)
    rows = []
    for number, (name, term) in enumerate(cells, 1):
        dataset, variates = gift_dataset(Dataset, name, term)
        horizon, meta = dataset.prediction_length, gift_metadata(properties, name, variates)
        rng = random.Random(stable_seed(seed, name, term))
        series = list(dataset.training_dataset)
        wanted = set(rng.sample(range(len(series)), min(len(series), 2 * rows_per_cell))) \
            if len(series) >= rows_per_cell else set(range(len(series)))
        draws = 1 if len(series) >= rows_per_cell else math.ceil(rows_per_cell / max(len(series), 1))
        candidates = []
        for index, entry in enumerate(series):
            if index not in wanted:
                continue
            target = np.asarray(entry["target"], np.float32)
            latest = len(target) - horizon
            if latest < 4:
                continue
            cutoffs = {latest}
            while len(cutoffs) < min(draws, latest - 3):
                cutoffs.add(rng.randint(4, latest))
            for cutoff in cutoffs:
                begin = max(0, cutoff - max_history)
                history = repair(target[begin:cutoff])
                if len(history) < 4:
                    history = np.pad(history, (4 - len(history), 0), mode="edge")
                future = repair(target[cutoff:cutoff + horizon])
                candidates.append(dict(
                    history=history.tolist(), future=future.tolist(), freq=str(dataset.freq), term=term,
                    dataset=name, source_series=index, source_group=group_of("GIFT", name.split("/")[0], index),
                    cutoff=cutoff, history_start=timestamp(entry["start"] + begin), **meta))
        rng.shuffle(candidates)
        candidates = candidates[:rows_per_cell]
        held = min(round(rows_per_cell * validation_fraction), max(1, round(len(candidates) * validation_fraction)))
        unique = candidates[held:]
        train = [unique[i % len(unique)] for i in range(rows_per_cell - round(rows_per_cell * validation_fraction))]
        random.Random(stable_seed(seed + 1, name, term)).shuffle(train)
        rows += train
        log(block="gift_sampled", cell=f"{name}/{term}", done=number, cells=len(cells), rows=len(rows))
    return rows


def gift_rolling(root, data, *, rows_per_cell=900, max_history=MAX_HISTORY, seed=20260729, exclude=()):
    """Rolling origins, one horizon apart, at the end of each training split; windows (or
    histories) that also occur in ``exclude`` are dropped."""
    Dataset, cells, properties = gift_eval(root, data)
    seen_windows = {(r["dataset"], r["source_series"], r["cutoff"], r["term"]) for r in exclude}
    seen_histories = {history_digest(r["history"]) for r in exclude}
    rows = []
    for number, (name, term) in enumerate(cells, 1):
        dataset, variates = gift_dataset(Dataset, name, term)
        horizon, windows = dataset.prediction_length, dataset.windows
        meta = gift_metadata(properties, name, variates)
        rng = random.Random(stable_seed(seed, name, term, "validation"))
        series = list(dataset.training_dataset)
        refs, available, earlier = [], 0, False
        for index, entry in enumerate(series):  # reservoir sample of (series, cutoff)
            length = len(entry["target"])
            earlier |= length - horizon * (windows + 1) >= 4
            for lag in range(windows, 0, -1):
                cutoff = length - horizon * lag
                if cutoff < 4:
                    continue
                available += 1
                if len(refs) < rows_per_cell:
                    refs.append((index, cutoff))
                elif (slot := rng.randrange(available)) < rows_per_cell:
                    refs[slot] = (index, cutoff)
        if not (earlier and refs):  # too short for rolling blocks: hold out whole series instead
            refs = [(i, len(e["target"]) - horizon) for i, e in enumerate(series) if len(e["target"]) - horizon >= 4]
            rng.shuffle(refs)
            refs = refs[:min(rows_per_cell, max(1, len(refs) // 5), len(refs) - 1)]
        cell = []
        for index, cutoff in refs:
            target = np.asarray(series[index]["target"], np.float32)
            begin = max(0, cutoff - max_history)
            history = repair(target[begin:cutoff])
            if len(history) < 4:
                history = np.pad(history, (4 - len(history), 0), mode="edge")
            cell.append(dict(
                history=history.tolist(), future=repair(target[cutoff:cutoff + horizon]).tolist(),
                freq=str(dataset.freq), term=term, dataset=name, source_series=index,
                source_group=group_of("GIFT", name.split("/")[0], index), cutoff=cutoff,
                history_start=timestamp(series[index]["start"] + begin), **meta))
        rng.shuffle(cell)
        for row in cell:
            key = (row["dataset"], row["source_series"], row["cutoff"], row["term"])
            digest = history_digest(row["history"])
            if key not in seen_windows and digest not in seen_histories:
                rows.append(row)
            seen_windows.add(key)
            seen_histories.add(digest)
        log(block="gift_rolling", cell=f"{name}/{term}", done=number, cells=len(cells), rows=len(rows))
    return rows


# ---------------------------------------------------------------------- BOOM
def boom(snapshot, *, count=100_000, windows_per_channel=6, seed=20260815, download=False):
    import pyarrow.ipc as ipc

    snapshot = Path(snapshot)
    if download:
        from huggingface_hub import snapshot_download
        snapshot_download(BOOM["repo"], repo_type="dataset", revision=BOOM["revision"], local_dir=snapshot)
    taxonomy = json.loads((snapshot / "dataset_taxonomy.json").read_text())
    rows = []
    for number, root in enumerate(sorted(taxonomy), 1):
        (path,) = sorted((snapshot / root).glob("*.arrow"))
        with ipc.open_stream(str(path)) as reader:
            (source,) = [row for batch in reader for row in batch.to_pylist()]
        for channel, values in enumerate(channels(source["target"])):
            rows += sample_windows(values, frequency=str(source.get("freq") or "unknown"),
                                   dataset=f"BOOM/{root}", series_id=f"{root}/{channel}",
                                   group=group_of("BOOM", root), limit=windows_per_channel,
                                   max_history=MAX_HISTORY, seed=seed, domain=DOMAINS["web_cloudops"],
                                   source_family=root, start=source.get("start"))
        if number % 50 == 0:
            log(block="boom", roots=number, of=len(taxonomy), windows=len(rows))
    return balanced(clean(rows), count, seed)


# --------------------------------------------------------------------- RMISC
def _rmisc_columns(schema, meta, rule):
    import pyarrow as pa

    numeric = {f.name for f in schema if pa.types.is_integer(f.type) or pa.types.is_floating(f.type)}
    numeric -= {"_original_filename", str(meta.get("timestamp") or ""), *map(str, meta.get("covariates") or ())}
    if rule == "targets":
        columns = [str(c) for c in meta.get("targets") or ()]
    elif rule == "numeric":
        columns = sorted(numeric)
    else:
        columns = sorted(c for c in numeric if re.search(rule, c))
    if not columns or set(columns) - numeric:
        raise ValueError(f"no usable target columns ({rule})")
    return columns


def rmisc_windows(snapshot, root, frequency, domain, rule, *, per_series, limit, seed):
    """Windows from every target column of every original file, split at missing values."""
    import pyarrow.parquet as pq

    directory = Path(snapshot) / root
    meta = json.loads((directory / "meta.json").read_text())
    payloads = sorted(directory.glob("*.parquet"))
    columns = _rmisc_columns(pq.read_schema(payloads[0]), meta, rule)
    table = pq.read_table(payloads, columns=[*columns, "_original_filename"])
    files = np.asarray(table["_original_filename"].to_numpy(zero_copy_only=False)).astype(str)
    edges = np.flatnonzero(np.r_[True, files[1:] != files[:-1], True])
    rows = []
    for column in columns:
        values = np.asarray(table[column].to_numpy(zero_copy_only=False), dtype=np.float32)
        for begin, end in zip(edges[:-1], edges[1:]):
            local = values[begin:end]
            finite = np.r_[False, np.isfinite(local), False]
            changes = np.flatnonzero(finite[1:] != finite[:-1])
            for run, (start, stop) in enumerate(zip(changes[::2], changes[1::2])):
                if stop - start < 12:
                    continue
                rows += sample_windows(local[start:stop], frequency=frequency, dataset=f"RMISC/{root}",
                                       series_id=f"{files[begin]}/{column}/{run}",
                                       group=group_of("RMISC", root, files[begin]), limit=per_series,
                                       max_history=MAX_HISTORY, seed=seed, domain=DOMAINS[domain],
                                       source_family=root)
    random.Random(stable_seed(seed, root, "pool")).shuffle(rows)
    log(block="rmisc", root=root, windows=min(len(rows), limit))
    return rows[:limit]


def rmisc(snapshot, *, count=49_000, per_series=24, root_limit=30_000, seed=20260815, download=False):
    if download:
        _download_rmisc(snapshot, RMISC_ROOTS)
    rows = [w for root, spec in RMISC_ROOTS.items()
            for w in rmisc_windows(snapshot, root, *spec, per_series=per_series, limit=root_limit, seed=seed)]
    return balanced(clean(rows, drop_constant=False), count, seed)


def _download_rmisc(snapshot, roots):
    from huggingface_hub import snapshot_download
    snapshot_download(RMISC["repo"], repo_type="dataset", revision=RMISC["revision"], local_dir=snapshot,
                      allow_patterns=[f"{root}/*" for root in roots])


# --------------------------------------------------------------------- LOTSA
def lotsa(*, count=100_000, windows_per_channel=48, max_series=8192, seed=20260814, workers=8):
    """Balanced windows from LOTSA datasets of seven domains, streamed from the Hub."""
    from concurrent.futures import ThreadPoolExecutor

    target = math.ceil(count / len(LOTSA_DOMAINS))
    jobs = []
    for domain, families in LOTSA_DOMAINS.items():
        for family, names in families.items():
            limit = max(256, math.ceil(1.5 * math.ceil(2.0 * target / len(families)) / len(names)))
            jobs += [(name, domain, family, limit) for name in names]

    def load(job):
        from datasets import load_dataset

        name, domain, family, limit = job
        rows = []
        stream = load_dataset(LOTSA["repo"], name, split="train", revision=LOTSA["revision"],
                              streaming=True, trust_remote_code=True)
        for index, source in enumerate(stream):
            if index >= max_series or len(rows) >= limit:
                break
            item = str(source.get("item_id") or index)
            for channel, values in enumerate(channels(source["target"])):
                rows += sample_windows(values, frequency=str(source.get("freq")), dataset=name,
                                       series_id=f"{item}/{channel}", group=group_of("LOTSA", name, item),
                                       limit=windows_per_channel, max_history=MAX_HISTORY, seed=seed,
                                       domain=DOMAINS[domain], source_family=family, start=source.get("start"))
        random.Random(stable_seed(seed, name, "rows")).shuffle(rows)
        log(block="lotsa", dataset=name, windows=min(len(rows), limit))
        return rows[:limit]

    with ThreadPoolExecutor(workers) as pool:
        rows = [w for part in pool.map(load, jobs) for w in part]
    return balanced(clean(rows, drop_constant=False), count, seed)


# ------------------------------------------------------------ UTSD + retail
def diverse(utsd_root, rmisc_snapshot, *, per_family=4000, retail=4000, windows_per_series=16, retail_per_series=24,
            seed=20260815, download=False):
    import pyarrow as pa

    utsd_root = Path(utsd_root)
    if download:
        from huggingface_hub import snapshot_download
        snapshot_download(UTSD["repo"], repo_type="dataset", revision=UTSD["revision"], local_dir=utsd_root,
                          allow_patterns=["UTSD-1G/*"])
        _download_rmisc(rmisc_snapshot, RETAIL_ROOTS)
    pools = defaultdict(list)
    for path in sorted((utsd_root / "UTSD-1G").glob("*.arrow")):
        with pa.memory_map(str(path)) as source:
            try:
                table = pa.ipc.open_stream(source).read_all()
            except pa.ArrowInvalid:
                table = pa.ipc.open_file(source).read_all()
        for item, values in zip(table["item_id"].to_pylist(), table["target"].to_pylist(), strict=True):
            family = item.rsplit("_", 2)[0]
            if family not in UTSD_FAMILIES:
                continue
            frequency, domain = UTSD_FAMILIES[family]
            pools[family] += sample_windows(np.asarray(values, np.float32), frequency=frequency,
                                            dataset=f"UTSD/{family}", series_id=item,
                                            group=group_of("UTSD", family, item), limit=windows_per_series,
                                            max_history=MAX_HISTORY, seed=seed, domain=DOMAINS[domain],
                                            source_family=family)
        log(block="diverse", shard=path.name, windows=sum(map(len, pools.values())))
    rows = []
    for family in sorted(pools):
        members = clean(pools[family], drop_constant=False)
        random.Random(stable_seed(seed, family)).shuffle(members)
        rows += members[:per_family]
    for root, spec in RETAIL_ROOTS.items():
        members = clean(rmisc_windows(rmisc_snapshot, root, *spec, per_series=retail_per_series,
                                      limit=10 * retail, seed=seed), drop_constant=False)
        random.Random(stable_seed(seed, root)).shuffle(members)
        rows += members[:retail]
    return rows


# ------------------------------------------------------------------------ CLI
def build(args) -> dict:
    output = Path(args.output)
    blocks = output / "blocks"
    counts = {}

    def block(name, make):
        path = blocks / f"{name}.jsonl"
        if not path.exists():
            rows = make()
            for row in rows:
                row["block"] = name
            write_jsonl(path, rows)
        counts[name] = sum(1 for _ in path.open())
        log(block=name, complete=counts[name])
        return path

    sampled = block("gift_sampled", lambda: gift_sampled(args.gift_root, args.gift_data))
    block("gift_rolling", lambda: gift_rolling(args.gift_root, args.gift_data,
                                               exclude=[json.loads(line) for line in sampled.open()]))
    block("boom", lambda: boom(args.boom, download=args.download))
    block("rmisc", lambda: rmisc(args.rmisc, download=args.download))
    block("lotsa", lambda: lotsa(workers=args.workers))
    block("diverse", lambda: diverse(args.utsd, args.rmisc, download=args.download))
    manifest = dict(blocks={name: dict(path=f"blocks/{name}.jsonl", rows=counts[name]) for name in BLOCKS})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gift-root", default="third_party/gift-eval", help="GIFT-Eval checkout")
    parser.add_argument("--gift-data", default="data/gift-eval", help="Salesforce/GiftEval dataset directory")
    parser.add_argument("--boom", default="data/sources/boom")
    parser.add_argument("--rmisc", default="data/sources/rmisc")
    parser.add_argument("--utsd", default="data/sources/utsd")
    parser.add_argument("--output", default="data/aggregation")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--download", action="store_true", help="download BOOM, RMISC and UTSD first")
    print(json.dumps(build(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
