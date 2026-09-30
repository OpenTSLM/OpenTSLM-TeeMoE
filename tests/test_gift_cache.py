import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from teemoe.eval import gift
from teemoe.model.mixture import EXPERTS


@pytest.fixture
def args(tmp_path, monkeypatch):
    for name in ("TEEMOE_MODEL_CLASS", "TEEMOE_MODEL_OPTIONS", "TEEMOE_VLLM_PYTHON"):
        monkeypatch.delenv(name, raising=False)
    checkpoint = tmp_path / "checkpoint"
    files = [checkpoint / name for name in
             ("teemoe.json", "editor.safetensors", "controller.safetensors", "router.ubj")]
    files += [checkpoint / "adapters" / expert / name for expert in EXPERTS
              for name in ("adapter_config.json", "adapter_model.safetensors")]
    for file in files:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"test")
    return SimpleNamespace(checkpoint=str(checkpoint), output=str(tmp_path / "output"), gpus=["0", "1"],
                           backend="vllm", vllm_python=None, environments=None, gift_root=str(tmp_path / "gift"),
                           gift_data=str(tmp_path / "data"), fnf_root=None, granite=None)


def test_gift_resumes_matching_run(args):
    from pathlib import Path

    gift.prepare_run(args)
    saved = Path(args.output) / "settings.json"
    before = saved.read_bytes()
    gift.prepare_run(args)
    assert saved.read_bytes() == before
    assert len(json.loads(before)["checkpoint_files"]) == 10


@pytest.mark.parametrize("change", ["checkpoint", "weights", "gpus", "backend", "data", "ablation"])
def test_gift_refuses_incompatible_cache(args, monkeypatch, change):
    from pathlib import Path
    import shutil

    gift.prepare_run(args)
    output = Path(args.output)
    sentinel = output / "report.json"
    sentinel.write_text("keep these results")
    if change == "checkpoint":
        copied = Path(args.checkpoint).with_name("other-checkpoint")
        shutil.copytree(args.checkpoint, copied)
        args.checkpoint = str(copied)
    elif change == "weights":
        weights = Path(args.checkpoint) / "controller.safetensors"
        original = weights.stat()
        weights.write_bytes(b"edit")  # same path and size, new contents
        os.utime(weights, ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000_000))
    elif change == "gpus":
        args.gpus.append("2")
    elif change == "backend":
        args.backend = "transformers"
    elif change == "data":
        args.gift_data += "-other"
    else:
        monkeypatch.setenv("TEEMOE_MODEL_OPTIONS", '{"weights": [1, 1, 1]}')
    with pytest.raises(ValueError, match="different checkpoint or evaluation configuration"):
        gift.prepare_run(args)
    assert sentinel.read_text() == "keep these results"


def test_gift_refuses_legacy_unbound_output(args):
    from pathlib import Path

    output = Path(args.output)
    (output / "run").mkdir(parents=True)
    (output / "run" / "worker00-0000.npz").write_bytes(b"old results")
    with pytest.raises(ValueError, match="Use a new --output"):
        gift.prepare_run(args)


@pytest.fixture
def scoring(tmp_path, monkeypatch):
    class Panel:
        cells = ["demo/H/short"]
        meta = [dict(cell=0), dict(cell=0)]
        scale = np.ones(2)

        def __init__(self, path):
            pass

        def __len__(self):
            return 2

        def target(self, index):
            return np.ones(3)

    monkeypatch.setattr(gift, "Panel", Panel)
    monkeypatch.setattr(gift, "leaderboard", lambda root: {"peer": {"demo/H/short": 2.0}})
    (tmp_path / "run").mkdir()

    def part(number, indices, **changes):
        payload = dict(index=np.asarray(indices, dtype=int), numerical=np.ones(len(indices), bool),
                       quantiles=np.ones((3 * len(indices), 9)) * 2)
        np.savez(tmp_path / "run" / f"worker{number:02d}-0000.npz", **dict(payload, **changes))

    return SimpleNamespace(output=str(tmp_path), gift_root=str(tmp_path)), part


def test_gift_scores_complete_unique_shards(scoring):
    from pathlib import Path

    args, part = scoring
    part(0, [0])
    part(1, [1])
    (Path(args.output) / "run" / "worker02-0000.tmp.npz").write_bytes(b"incomplete write")
    assert gift.score(args)["windows"] == 2


@pytest.mark.parametrize("indices", [[0], [1, 1], [-1], [2]])
def test_gift_rejects_duplicate_or_invalid_indices(scoring, indices):
    args, part = scoring
    part(0, [0])
    part(1, indices)
    with pytest.raises(ValueError, match="Duplicate or invalid"):
        gift.score(args)


def test_gift_rejects_incomplete_forecasts(scoring):
    args, part = scoring
    part(0, [0, 1], quantiles=np.ones((3, 9)))
    with pytest.raises(ValueError, match="Incomplete GIFT"):
        gift.score(args)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_gift_rejects_nonfinite_forecasts(scoring, value):
    args, part = scoring
    quantiles = np.ones((6, 9))
    quantiles[0, 4] = value
    part(0, [0, 1], quantiles=quantiles)
    with pytest.raises(ValueError, match="Nonfinite GIFT forecasts"):
        gift.score(args)


def test_gift_rejects_cells_without_valid_mase(scoring, monkeypatch):
    args, part = scoring
    part(0, [0, 1])
    monkeypatch.setattr(gift.Panel, "scale", np.full(2, np.nan))
    with pytest.raises(ValueError, match="valid MASE observations"):
        gift.score(args)


@pytest.mark.parametrize("peers", [{}, {"peer": {"demo/H/short": np.nan}}])
def test_gift_rejects_missing_or_invalid_leaderboard(scoring, monkeypatch, peers):
    args, part = scoring
    part(0, [0, 1])
    monkeypatch.setattr(gift, "leaderboard", lambda root: peers)
    with pytest.raises(ValueError, match="leaderboard"):
        gift.score(args)
