import numpy as np

from teemoe.ensemble.fnf import blend, canonical_frequency, fit_blend_mass
from teemoe.ensemble.router import allocations, crps_ranks, expert_features, pool_quantiles, router_features


def test_forecaster_environment_isolation(monkeypatch, tmp_path):
    from teemoe.ensemble import forecasters

    folders = {"default": ".venv-fm", "timer_s1": ".venv-timer", "moirai2": ".venv-moirai",
               "tirex2": ".venv-tirex"}
    for folder in folders.values():
        python = tmp_path / folder / "bin/python"
        python.parent.mkdir(parents=True)
        python.touch()
    monkeypatch.setattr(forecasters, "REPO_ROOT", tmp_path)
    assert forecasters.default_environments() == {
        key: str(tmp_path / folder / "bin/python") for key, folder in folders.items()}


def test_router_inputs_and_expansion():
    rng = np.random.default_rng(0)
    history = 10 + np.sin(np.arange(200) / 5)
    quantiles = np.sort(rng.normal(10, 1, (8, 24, 9)), axis=-1)
    features = router_features(history, quantiles, frequency="H", term="short")
    assert features.shape == (195,) and np.isfinite(features).all()
    assert expert_features(np.stack([features] * 3)).shape == (24, 1433)
    weights = allocations(rng.normal(size=16))
    assert weights.shape == (2, 8) and np.allclose(weights.sum(1), 1)


def test_pooling_and_ranks():
    q = np.sort(np.random.default_rng(1).normal(size=(1, 1, 6, 9)), axis=-1)
    same = np.repeat(q, 8, axis=1)
    assert np.allclose(pool_quantiles(same, np.full((1, 8), 1 / 8))[0], q[0, 0])
    target = q[0, 0, :, 4][None]
    shifted = same + np.arange(8)[None, :, None, None]
    assert list(crps_ranks(shifted, target, np.ones(1))[0]) == list(range(8))


def test_blend_and_mass():
    left, right = np.zeros((4, 9), np.float32), np.ones((4, 9), np.float32)
    assert np.allclose(blend(left, right, 0.25), 0.25)
    assert np.allclose(blend(left, np.full_like(right, np.nan), 0.25), 0)
    target = np.full((3, 4), 0.4)
    assert abs(fit_blend_mass(left[:, 4][None].repeat(3, 0), right[:, 4][None].repeat(3, 0), target) - 0.4) < 1e-9
    assert canonical_frequency("1h") == "H" and canonical_frequency("15min") == "15T"
