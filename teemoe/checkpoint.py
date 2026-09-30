"""The TeeMoE checkpoint layout, and packaging the training outputs into it:

    python -m teemoe.checkpoint --aggregation artifacts/aggregation --native artifacts/native \
        --analysis artifacts/analysis --controller artifacts/controller --output checkpoints/teemoe


    teemoe.json                   base model, expert order, Toto-FnF blend weight
    adapters/aggregation/         PEFT LoRA (rank 4)
    adapters/native/              PEFT LoRA (rank 32)
    adapters/analysis/            PEFT LoRA (rank 16)
    editor.safetensors            numerical connector and forecast decoder
    controller.safetensors        expert controller
    router.ubj                    XGBoost weighting of the 8 core forecasters
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .model.backbone import BASE_MODEL, BASE_REVISION
from .model.mixture import EXPERTS

DEFAULT_REPO = "OpenTSLM/TeeMoE"


def resolve(path_or_repo: str | Path = DEFAULT_REPO, revision: str | None = None) -> Path:
    path = Path(path_or_repo)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(path_or_repo), revision=revision))


def save_checkpoint(destination: str | Path, *, adapters: dict[str, str | Path],
                    editor: dict[str, torch.Tensor], controller: dict[str, torch.Tensor],
                    router: str | Path, fnf_mass: float, base_model: str = BASE_MODEL,
                    base_revision: str = BASE_REVISION) -> Path:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in EXPERTS:
        target = destination / "adapters" / name
        target.mkdir(parents=True, exist_ok=True)
        source = Path(adapters[name])
        shutil.copyfile(source / "adapter_model.safetensors", target / "adapter_model.safetensors")
        config = json.loads((source / "adapter_config.json").read_text())
        config.update(base_model_name_or_path=base_model, revision=base_revision)
        (target / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file({k: v.detach().cpu().contiguous() for k, v in editor.items()}, str(destination / "editor.safetensors"))
    save_file({k: v.detach().cpu().contiguous() for k, v in controller.items()},
              str(destination / "controller.safetensors"))
    shutil.copyfile(router, destination / "router.ubj")
    (destination / "teemoe.json").write_text(json.dumps(dict(
        base_model=base_model, base_revision=base_revision, experts=list(EXPERTS),
        fnf_mass=float(fnf_mass)), indent=2) + "\n")
    return destination


def read_config(root: Path) -> dict:
    config = json.loads((root / "teemoe.json").read_text())
    if tuple(config["experts"]) != EXPERTS:
        raise ValueError("unexpected expert order in teemoe.json")
    return config


def adapter_dirs(root: Path) -> list[Path]:
    return [root / "adapters" / name for name in EXPERTS]


def load_state(path: Path) -> dict[str, torch.Tensor]:
    return load_file(str(path))


def package(aggregation: Path, native: Path, analysis: Path, controller: Path, output: Path) -> Path:
    """Assemble a checkpoint from the outputs of the four training commands."""
    return save_checkpoint(
        output, adapters=dict(aggregation=aggregation / "expert/adapter", native=native / "adapter",
                              analysis=analysis / "adapter"),
        editor=load_file(str(aggregation / "expert/editor.safetensors")),
        controller=load_file(str(controller / "controller.safetensors")), router=aggregation / "routers/full.ubj",
        fnf_mass=json.loads((aggregation / "blend.json").read_text())["fnf_mass"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, default in (("aggregation", "artifacts/aggregation"), ("native", "artifacts/native"),
                          ("analysis", "artifacts/analysis"), ("controller", "artifacts/controller"),
                          ("output", "checkpoints/teemoe")):
        parser.add_argument(f"--{name}", type=Path, default=Path(default))
    args = parser.parse_args()
    print(package(args.aggregation, args.native, args.analysis, args.controller, args.output))


if __name__ == "__main__":
    main()
