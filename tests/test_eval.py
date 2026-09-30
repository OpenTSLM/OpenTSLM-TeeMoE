import numpy as np

from teemoe.eval.cik import stable_crps
from teemoe.teemoe import inverse_cdf, sample_quantiles


def test_stable_crps_matches_the_definition():
    rng = np.random.default_rng(0)
    samples, target = rng.normal(size=(25, 7)), rng.normal(size=7)
    n = len(samples)
    naive = np.abs(samples - target).mean(0) - np.abs(samples[:, None] - samples[None]).sum((0, 1)) / (2 * n * (n - 1))
    assert np.allclose(stable_crps(target, samples), naive)


def test_sampling_helpers():
    quantiles = np.sort(np.random.default_rng(1).normal(size=(5, 9)), axis=-1)
    draws = inverse_cdf(quantiles, 1000, seed=0)
    assert draws.shape == (1000, 5)
    assert (draws >= quantiles[:, :1].T).all() and (draws <= quantiles[:, -1:].T).all()
    assert sample_quantiles(np.arange(25)[:, None] * np.ones((1, 3))).shape == (3, 9)
