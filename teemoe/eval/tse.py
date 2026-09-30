"""TimeSeriesExam v1.1: accuracy over its 746 multiple-choice questions.

    python -m teemoe.eval.tse --checkpoint OpenTSLM/TeeMoE --output results/tse.json

Each question is answered greedily (at most 64 tokens) with the authors' concept
descriptions and question hints supplied. An answer is correct when it contains
the correct ``LETTER) option`` line.
"""

from __future__ import annotations

import argparse
import ast
import json
from urllib.request import urlopen

from .common import add_model_arguments, load_model, save_report

DATA = dict(repo="AutonLab/TimeSeriesExam1", revision="628f686757bb484d9b608b6855fb41f6a302ecda",
            file="data/test-00000-of-00001.parquet")
CONCEPTS = ("https://raw.githubusercontent.com/moment-timeseries-foundation-model/TimeSeriesExam/"
            "384cf50864860c65e962b441eaa4c201857a06f8/evaluate/concepts.py")


def load_questions() -> tuple[list[dict], dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATA["repo"], DATA["file"], repo_type="dataset", revision=DATA["revision"])
    samples = [{k: v for k, v in row.items() if v is not None} for row in pq.read_table(path).to_pylist()]
    with urlopen(CONCEPTS, timeout=60) as response:  # read the literal table; never executed
        tree = ast.parse(response.read().decode())
    table = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                 and any(getattr(t, "id", None) == "CONCEPTS" for t in node.targets))
    concepts = {}
    for key, value in zip(table.keys, table.values, strict=True):
        fields = {k.arg: k.value for k in value.keywords}
        concepts[ast.literal_eval(key)] = tuple(ast.literal_eval(fields[f]) for f in ("concept_name",
                                                                                      "concept_description"))
    return samples, concepts


def request(sample: dict, concepts: dict) -> dict:
    series = [sample["ts"]] if "ts" in sample else [sample["ts1"], sample["ts2"]]
    terms = [concepts[key] for key in sample.get("relevant_concepts", [])[:3]]
    return dict(series=series, question=str(sample["question"]), options=list(sample["options"]),
                concept_definitions="\n".join(f"{name}: {description}." for name, description in terms),
                clarification=str(sample.get("question_hint") or ""))


def target(sample: dict) -> str:
    return f"{chr(65 + list(sample['options']).index(sample['answer']))}) {sample['answer']}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser)
    parser.add_argument("--output", default="results/tse.json")
    args = parser.parse_args()
    samples, concepts = load_questions()
    model = load_model(args)
    answers = model.analyze_batch([request(sample, concepts) for sample in samples], max_tokens=64)
    records = [dict(index=i, answer=target(s), response=a.text, weights=a.weights.tolist(),
                    correct=target(s).casefold() in a.text.casefold())
               for i, (s, a) in enumerate(zip(samples, answers, strict=True))]
    accuracy = sum(r["correct"] for r in records) / len(records)
    save_report(args.output, dict(benchmark="TimeSeriesExam v1.1", questions=len(records), accuracy=accuracy,
                                  records=records))
    print(json.dumps(dict(questions=len(records), accuracy=accuracy)))


if __name__ == "__main__":
    main()
