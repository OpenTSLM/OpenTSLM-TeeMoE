"""Training examples for the time-series analysis expert.

    python -m teemoe.data.analysis --chatts data/sources/chatts/uts_template_512_3000_no.jsonl \
        --output data/analysis/train.jsonl

  source                                                        examples
  Time-MQA/TSQA: activity classification, anomaly detection      2,000 + 2,000
  ChengsenWang/TSQA multiple choice                               3,000
  HiTSR: level-1 descriptions, level-2 multiple choice              2,000
  ChatTS author generator (univariate templates)                   3,000

Questions and answers are the authors'; the evidence is the shared analysis
rendering (raw values and fixed measurements). Time-MQA/TSQA is gated on the
Hugging Face Hub: accept its terms and run ``hf auth login`` first. The ChatTS
examples come from the authors' generator (checked out by ``scripts/setup.sh``).
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import re
from pathlib import Path

import numpy as np

from ..evidence import enrich_source_evidence, sketch
from ..prompts import CHOICE_SCHEMA, FREEFORM_SCHEMA, render_user
from .windows import log, write_jsonl

ASSETS = {  # name: (repository, revision, file)
    "chengsen": ("ChengsenWang/TSQA", "5f9565d441f7e5fa6c3635f0686386065c662f6d", "TSQA.csv"),
    "time_mqa_anomaly": ("Time-MQA/TSQA", "a7f9e95667b695d2647b9eff92bb6a38b5183a34",
                         "Anomaly_Detection/anomaly_detection.csv"),
    "time_mqa_classification": ("Time-MQA/TSQA", "a7f9e95667b695d2647b9eff92bb6a38b5183a34",
                                "Classification/classification.csv"),
    "hitsr_l1": ("November-Rain/HiTSR", "566be99bf403051df326eeb883d4e3888cbe508b", "Train/l1_train.json"),
    "hitsr_l2": ("November-Rain/HiTSR", "566be99bf403051df326eeb883d4e3888cbe508b", "Train/l2_train.json"),
}
PATTERN_QUESTION = ("Given the time series data, select the description from the four options that best "
                    "corresponds to the provided data.")


def number(value) -> str:
    return format(float(value), ".4g")


def series_evidence(values) -> str:
    return ("Time series values (successive observations separated by spaces):\n"
            + " ".join(number(x) for x in values) + "\nComputed measurements:\n" + sketch({"ts": values}))


def sensor_evidence(values: list[float], question: str) -> str:
    """Time-MQA activity data as the authors lay it out (XYZ readings, or nine sensor values)."""
    if "each timestamp containing X, Y, and Z values" in question and len(values) == 30:
        lines = ["Accelerometer observations at successive times (20 Hz).", "Observation | X | Y | Z"]
        lines += [f"{t} | " + " | ".join(number(x) for x in values[3 * t:3 * t + 3]) for t in range(10)]
        lines.append("Computed measurements for each temporal channel separately:")
        lines += [f"{axis} channel:\n" + sketch({"ts": values[i::3]}) for i, axis in enumerate("XYZ")]
        return "\n".join(lines)
    if "nine measurements (following the order)" in question and len(values) == 9:
        lines = ["Sensor measurements (mg), not a sequence of nine times:",
                 "Location | Horizontal forward | Vertical | Horizontal lateral"]
        lines += [place + " | " + " | ".join(number(x) for x in values[3 * i:3 * i + 3])
                  for i, place in enumerate(("Ankle (lower leg)", "Thigh (above the knee)", "Hip"))]
        return "\n".join(lines)
    raise ValueError("unrecognized sensor layout")


def finish(row: dict, values) -> dict:
    row["evidence"] = enrich_source_evidence(row["evidence"], values)
    render_user(row)
    return row


def freeform(row_id: str, values, question: str, answer: str, evidence: str | None = None) -> dict:
    return finish(dict(row_id=row_id, mold="freeform_analysis", evidence=evidence or series_evidence(values),
                       question=question, target=answer, output_schema=FREEFORM_SCHEMA), values)


def choice(row_id: str, values, question: str, options: list[tuple[str, str]], answer: str) -> dict:
    return finish(dict(row_id=row_id, mold="abcd_analysis", evidence=series_evidence(values), question=question,
                       options="\n".join(f"{label}) {text}" for label, text in options),
                       target=f"{answer}) {dict(options)[answer]}", output_schema=CHOICE_SCHEMA), values)


# ---------------------------------------------------------------- converters
def time_mqa(task: str, index: int, raw: dict) -> dict:
    qa = json.loads("{" + raw["QA_list"] + "}")
    (match,) = re.finditer(r"\[(?:\s*[-+0-9.eE]+\s*,?)+\]", qa["question"])
    values = [float(x) for x in json.loads(match.group(0))]
    question = (qa["question"][:match.start()] + "<SERIES>" + qa["question"][match.end():]).strip()
    question = question.replace("<SERIES>", "the supplied time-series evidence")
    evidence = sensor_evidence(values, question) if task == "classification" else None
    return freeform(f"time_mqa::{task}:{index}", values, question, qa["answer"].strip(), evidence)


def chengsen(index: int, raw: dict) -> dict:
    values = [float(x) for x in ast.literal_eval(raw["Series"])]
    prompt = raw["Question"].strip()
    options = re.findall(r"^\(([a-c])\) (.+)$", prompt, re.M)
    answer = dict(options)[raw["Answer"].strip()[1]]

    def meaning(text: str) -> str:
        text = text.lower().replace("this time series", "the time series")
        text = text.replace("increasing", "increased").replace("decreasing", "decreased")
        return re.sub(r"\b(?:a|an)\s+", "", text).strip()

    (label,) = [label for label, text in options if meaning(text) == meaning(answer)]
    return choice(f"chengsen:{index}", values, prompt.split("\n(a)", 1)[0],
                  [(label.upper(), text) for label, text in options], label.upper())


def hitsr_l1(raw: dict) -> dict:
    (values,) = raw["timeseries"] if isinstance(raw["timeseries"], list) else json.loads(raw["timeseries"])
    return freeform(f"hitsr_l1:{raw['id']}", values, str(raw["prompt_1"]).strip(), str(raw["answer_1"]).strip())


def hitsr_l2(raw: dict, *, from_explanation: bool) -> dict:
    """A four-option description question; the answer from the authors' explanation or truth label."""
    (values,) = raw["timeseries"] if isinstance(raw["timeseries"], list) else json.loads(raw["timeseries"])
    if from_explanation:
        question = (str(raw["prompt_1"]) + str(raw["option"])).strip()
        options = re.findall(r"^([A-D]): (.+)$", question.split("\n\nOptions:\n", 1)[1], re.M)
        (answer,) = re.findall(r"the correct answer is ([A-D])\.", raw["answer_1"], re.I)
    else:
        options = [(label, text.strip()) for label, text in re.findall(
            r"(?:^|\n|Options:\s*)([A-D])[:.]\s*(.*?)(?=\n[A-D][:.]\s|$)", raw["option"], re.S)]
        answer = raw["truth"]
    if len(options) != 4 or len({text.strip().casefold() for _, text in options}) != 4:
        raise ValueError("ambiguous options")
    return choice(f"hitsr_l2:{raw['id']}", values, PATTERN_QUESTION, options, answer)


def chatts(index: int, raw: dict) -> dict:
    values = np.asarray(raw["timeseries"], dtype=float)[0]
    question = raw["input"].removeprefix("There is a time series of length 512: <ts><ts/>. ")
    answer = re.sub(r"np\.float64\(([-+\d.eE]+)\)", r"\1", raw["output"])
    return freeform(f"chatts:{index}", values, question, answer)


# ---------------------------------------------------------------- selection
def sample(rows, count: int, convert, seed: int, name: str) -> list[dict]:
    """``count`` examples from a random order of the source rows (skipping unusable ones)."""
    order = list(range(len(rows)))
    random.Random(f"{seed}:{name}").shuffle(order)
    result = []
    for index in order:
        try:
            result.append(convert(index, rows[index]))
        except (ValueError, KeyError, IndexError, SyntaxError, json.JSONDecodeError):
            continue
        if len(result) == count:
            break
    log(source=name, examples=len(result))
    return result


def read(path: Path):
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.open(encoding="utf-8")]
    return json.loads(path.read_text(encoding="utf-8"))


def build(assets: dict[str, Path], chatts_path: Path, seed: int = 0) -> list[dict]:
    rows = []
    for task in ("classification", "anomaly"):
        rows += sample(read(assets[f"time_mqa_{task}"]), 2000, lambda i, r, t=task: time_mqa(t, i, r), seed,
                       f"time_mqa_{task}")
    rows += sample(read(assets["chengsen"]), 3000, chengsen, seed, "chengsen")
    rows += sample(read(assets["hitsr_l1"]), 1087, lambda i, r: hitsr_l1(r), seed, "hitsr_l1")
    level2 = read(assets["hitsr_l2"])
    correct = lambda r: dict(re.findall(r"(?:^|\n)([A-D])[:.]\s*(.*)", r["option"])).get(r["truth"], "")
    periods = [r for r in level2 if "period length" in correct(r)]
    spikes = [r for r in level2 if correct(r).startswith("Spikes occur at")]
    used = {id(r) for r in periods + spikes}
    rows += sample([r for r in level2 if id(r) not in used], 113,
                   lambda i, r: hitsr_l2(r, from_explanation=True), seed, "hitsr_l2")
    rows += sample(periods, 615, lambda i, r: hitsr_l2(r, from_explanation=False), seed, "hitsr_periods")
    rows += sample(spikes, 185, lambda i, r: hitsr_l2(r, from_explanation=False), seed, "hitsr_spikes")
    rows += [chatts(i, r) for i, r in enumerate(read(chatts_path)[:3000])]
    random.Random(seed).shuffle(rows)
    return rows


CHATTS = dict(repository="https://github.com/NetManAIOps/ChatTS.git",
              revision="a16ca1a7bd2d0cbe1dd40af37cb5658e7008357e")


def generate_chatts(repo: str | Path, output: str | Path, questions: int = 3000) -> Path:
    """Run the ChatTS authors' univariate template generator (length 512, seed 1)."""
    import os
    import runpy
    import subprocess
    import sys
    import tempfile

    import yaml

    repo, output = Path(repo).resolve(), Path(output).resolve()
    config = yaml.safe_load(subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{CHATTS['revision']}:config/datagen_config.yaml"], text=True))
    config.update(seq_len=512, encoding_method="no", num_data_template_qa=questions, data_output_dir=str(output))
    cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as work:
        (Path(work) / "config").mkdir()
        (Path(work) / "config/datagen_config.yaml").write_text(yaml.safe_dump(config))
        sys.path.insert(0, str(repo))
        random.seed(1)
        np.random.seed(1)
        try:
            os.chdir(work)
            runpy.run_module("chatts.align.uts_template_qa", run_name="__main__")
        finally:
            os.chdir(cwd)
    return output / f"uts_template_512_{questions}_no.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", default="data/sources/analysis", help="download cache")
    parser.add_argument("--chatts", help="output of the ChatTS author generator (else it is generated)")
    parser.add_argument("--chatts-repo", default="third_party/ChatTS", help="ChatTS checkout")
    parser.add_argument("--asset", action="append", default=[], metavar="NAME=PATH",
                        help="use an already downloaded source file")
    parser.add_argument("--output", default="data/analysis/train.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download

    assets = dict(item.split("=", 1) for item in args.asset)
    for name, (repo, revision, filename) in ASSETS.items():
        if name not in assets:
            assets[name] = hf_hub_download(repo, filename, repo_type="dataset", revision=revision,
                                           local_dir=Path(args.sources) / name)
    chatts_path = args.chatts or generate_chatts(args.chatts_repo, Path(args.sources) / "chatts")
    rows = build({k: Path(v) for k, v in assets.items()}, Path(chatts_path), args.seed)
    print(json.dumps(dict(output=args.output, examples=write_jsonl(args.output, rows))))


if __name__ == "__main__":
    main()
