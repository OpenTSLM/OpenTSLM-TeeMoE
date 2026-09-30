import json

import torch
from safetensors.torch import load_file

from teemoe.checkpoint import save_checkpoint
from teemoe.model.backbone import BASE_MODEL, BASE_REVISION
from teemoe.model.mixture import EXPERTS


def test_checkpoint_packaging_uses_portable_base_model_names(experts, tmp_path):
    for source in experts:
        config = json.loads((source / "adapter_config.json").read_text())
        config["base_model_name_or_path"] = str(tmp_path / "local-model-cache")
        (source / "adapter_config.json").write_text(json.dumps(config))
    router = tmp_path / "router.ubj"
    router.write_bytes(b"router fixture")
    output = save_checkpoint(tmp_path / "package", adapters=dict(zip(EXPERTS, experts, strict=True)),
                             editor={"weight": torch.ones(2)}, controller={"weight": torch.zeros(2)},
                             router=router, fnf_mass=0.3)
    for name, source in zip(EXPERTS, experts, strict=True):
        target = output / "adapters" / name
        config = json.loads((target / "adapter_config.json").read_text())
        assert config["base_model_name_or_path"] == BASE_MODEL and config["revision"] == BASE_REVISION
        filename = "adapter_model.safetensors"
        assert (target / filename).read_bytes() == (source / filename).read_bytes()
    assert torch.equal(load_file(output / "editor.safetensors")["weight"], torch.ones(2))
    assert torch.equal(load_file(output / "controller.safetensors")["weight"], torch.zeros(2))
    assert (output / "router.ubj").read_bytes() == router.read_bytes()
