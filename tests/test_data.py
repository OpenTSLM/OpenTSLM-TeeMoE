import numpy as np
import pytest

from teemoe.data.native import bounds_context, example
from teemoe.data.windows import balanced, clean, sample_windows
from teemoe.prompts import render_user


def test_caf_selection_uses_only_the_bundled_rows():
    from teemoe.data.native import caf_selection

    selected = caf_selection()
    assert len(selected) == len(set(selected)) == 2800
    assert set(caf_selection(seed=7)) == set(selected)
    assert set(caf_selection(count=20, seed=9)) <= set(selected)
    assert caf_selection(count=0) == []
    with pytest.raises(ValueError, match="CAF count"):
        caf_selection(count=2801)


@pytest.mark.parametrize("complete", [True, False])
def test_caf_reconstructs_only_selected_rows(monkeypatch, tmp_path, complete):
    import datasets
    import huggingface_hub
    import pyarrow as pa
    import pyarrow.parquet as pq

    from teemoe.data import native

    filename = "train-00000-of-00129.parquet"
    metadata = dict(dataset_name="demo", series_idx=0, target_column="target", start_idx=0,
                    past_timestamp=["2024-01-01", "2024-01-02"], future_timestamp=["2024-01-03"],
                    context="selected context")
    (tmp_path / "data").mkdir()
    pq.write_table(pa.Table.from_pylist([dict(metadata, context="not selected"), metadata]),
                   tmp_path / "data" / filename)
    monkeypatch.setattr(native, "caf_selection", lambda count, seed: [(filename, 1)])
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **kw: str(tmp_path))
    values = [1.0, 2.0, 3.0] if complete else [1.0]
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **kw: iter([dict(target=values)]))
    if complete:
        rows = native.caf(tmp_path, count=1)
        assert len(rows) == 1 and rows[0]["context"] == "selected context"
    else:
        with pytest.raises(ValueError, match="Incomplete CAF"):
            native.caf(tmp_path, count=1)


def test_native_preparation_does_not_reuse_old_caf_cache(monkeypatch, tmp_path):
    import json
    import sys

    from teemoe.data import native

    sources = tmp_path / "sources"
    sources.mkdir()
    for name in ("tadiff", "nws", "caf", "bounded", "lotsa"):
        (sources / f"{name}.jsonl").write_text(json.dumps(dict(source=name, row_id="old")) + "\n")
    monkeypatch.setattr(native, "caf", lambda *a, **kw: [dict(source="caf", row_id="selected")])
    monkeypatch.setattr(sys, "argv", ["native", "--output", str(tmp_path / "train.jsonl")])
    native.main()
    rows = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert [row["row_id"] for row in rows if row["source"] == "caf"] == ["selected"]
    assert json.loads((sources / "caf.jsonl").read_text())["row_id"] == "old"


def test_sampled_windows():
    values = np.sin(np.arange(5000) / 10) + 2
    rows = sample_windows(values, frequency="H", dataset="demo", series_id="s", group="g", limit=9,
                          max_history=8192, seed=0, start="2020-01-01", domain="Energy", source_family="demo")
    assert 0 < len(rows) <= 9
    for row in rows:
        assert len(row["future"]) in (48, 480, 720) and len(row["history"]) <= 16 * len(row["future"])
        assert row["history_start"].startswith("2020")
    assert len(clean(rows + rows)) == len(rows)
    assert len(balanced(rows, 3, seed=0)) == 3


def test_native_example_and_imposed_bounds():
    row = example("x", [1, 2, 3, 4], [5, 0, 9], ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
                  ["2024-01-05", "2024-01-06", "2024-01-07"], source="demo")
    assert render_user(row).endswith("Future timestamps:\n2024-01-05\n2024-01-06\n2024-01-07")
    bounded = bounds_context(row, seed=0)
    values = [float(v) for v in bounded["target"].split(", ")[1:] for v in [v.split(")")[0]]]
    assert "reported series will have" in bounded["context"] and len(values) == 3
    assert min(values) >= min(0, 1) and max(values) <= 9
