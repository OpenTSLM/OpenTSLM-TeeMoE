"""Convert a model trained with the paper's reproduction code into a TeeMoE checkpoint.

    python paper/convert_checkpoint.py --deployment artifacts/composition/fit/deployment.json \
        --router artifacts/aggregation_reproduction/routers/full/router.ubj \
        --blend artifacts/aggregation_reproduction/parents/full/ensemble.json \
        --output checkpoints/teemoe

The deployment names the three adapters, the aggregation checkpoint (numerical
connector and decoder) and the controller checkpoint; the router and blend files
are the fitted numerical ensemble. Tensors are copied unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teemoe.checkpoint import save_checkpoint  # noqa: E402
from teemoe.model.editor import AggregationEditor  # noqa: E402

CAPABILITIES = dict(aggregation="aggregation", native_forecasting="native", analysis="analysis")


def convert(deployment: Path, router: Path, blend: Path, output: Path) -> Path:
    manifest = json.loads(deployment.read_text())
    resolve = lambda value: Path(value) if Path(value).is_absolute() else (deployment.parent / value).resolve()
    adapters = {CAPABILITIES[e["capability"]]: resolve(e["adapter"]) for e in manifest["experts"]}
    state = torch.load(resolve(manifest["aggregation_checkpoint"]), map_location="cpu", weights_only=False)["state"]
    expected = set(AggregationEditor(8).state_dict())
    editor = {k.removeprefix("editor."): v for k, v in state.items()
              if k.startswith("editor.") and k.removeprefix("editor.") in expected}
    if set(editor) != expected:
        raise ValueError(f"editor tensors missing: {sorted(expected - set(editor))}")
    classifier = torch.load(resolve(manifest["controller"]), map_location="cpu", weights_only=False)["controller_state"]
    pick = lambda suffix: next((v for k, v in classifier.items() if k.endswith(suffix)), None)
    fraction = pick("signal_fraction")  # the learned task-text / full-request balance
    if fraction is None:  # a controller that reads one view only
        view = manifest.get("routing_view", "instruction_context")
        fraction = dict(instruction_context=torch.zeros(()), full=torch.ones(()))[view]
    controller = {"head.weight": pick("layers.0.weight"), "head.bias": pick("layers.0.bias"),
                  "signal_fraction": fraction.reshape(())}
    fnf_mass = float(json.loads(blend.read_text())["ensemble"]["right_mass"])
    return save_checkpoint(output, adapters=adapters, editor=editor, controller=controller, router=router,
                           fnf_mass=fnf_mass, base_model=manifest["model_id"], base_revision=manifest["model_revision"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("deployment", "router", "blend", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(convert(args.deployment, args.router, args.blend, args.output))


if __name__ == "__main__":
    main()
