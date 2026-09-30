"""Shared command-line options for the benchmark evaluations."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", default="OpenTSLM/TeeMoE", help="Hugging Face repository or local directory")
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="vllm",
                        help="text generation backend")
    parser.add_argument("--device", default="cuda:0", help="device of the backbone")
    parser.add_argument("--vllm-devices", help="GPUs for vLLM, e.g. '1' (default: the backbone's GPU)")
    parser.add_argument("--vllm-python", help="Python of the vLLM environment (or TEEMOE_VLLM_PYTHON)")
    parser.add_argument("--forecast-devices", nargs="+", help="GPUs for the forecasting models")
    parser.add_argument("--environments", help="JSON file: forecasting model or kind -> Python executable")
    parser.add_argument("--granite", help="granite-tsfm checkout (FlowState, PatchTST-FM)")
    parser.add_argument("--fnf-root", help="Toto-FnF bundle directory (default: downloaded from the Hub)")


def load_model(args: argparse.Namespace):
    """TeeMoE, or the subclass named by TEEMOE_MODEL_CLASS ("module:Class", with keyword
    arguments in TEEMOE_MODEL_OPTIONS as JSON), e.g. the variants in paper/."""
    from ..teemoe import TeeMoE

    model_class, options = TeeMoE, json.loads(os.environ.get("TEEMOE_MODEL_OPTIONS", "{}"))
    if spec := os.environ.get("TEEMOE_MODEL_CLASS"):
        module, _, name = spec.partition(":")
        model_class = getattr(importlib.import_module(module), name)
    environments = json.loads(Path(args.environments).read_text()) if args.environments else None
    return model_class.from_pretrained(
        args.checkpoint, **options, device=args.device, backend=args.backend, vllm_devices=args.vllm_devices,
        vllm_python=args.vllm_python, forecast_devices=tuple(args.forecast_devices) if args.forecast_devices else None,
        environments=environments, granite=args.granite, fnf_root=args.fnf_root)


def save_report(path: str | Path, report: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
