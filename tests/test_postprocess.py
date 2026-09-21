"""Tests for the posterior post-processing helpers and the sampler utilities
used by the Union2.1 notebook (proposal tuning, dispersed starts)."""

import numpy as np
import pytest

from cosmo_library.bayes import (
    Posterior, GaussianRandomWalk, dispersed_starts, scaled_proposal_cov,
)
from cosmo_library.postprocess import contour_levels_2d, credible_interval


def test_credible_interval_gaussian():
    x = np.random.default_rng(0).normal(2.0, 3.0, size=400_000)
    med, lo, hi = credible_interval(x, 0.68)
    assert med == pytest.approx(2.0, abs=0.02)
    assert lo == pytest.approx(2.0 - 3.0, abs=0.03)
    assert hi == pytest.approx(2.0 + 3.0, abs=0.03)


def test_credible_interval_rejects_bad_level():
    with pytest.raises(ValueError):
        credible_interval([1.0, 2.0], level=1.5)


def test_contour_levels_enclose_requested_mass():
    rng = np.random.default_rng(1)
    s = rng.multivariate_normal([0, 0], [[1.0, 0.6], [0.6, 2.0]], size=400_000)
    xc, yc, dens, thr = contour_levels_2d(s[:, 0], s[:, 1], (0.68, 0.95))
    assert dens.shape == (yc.size, xc.size)
    assert thr[1] < thr[0]                      # 95% region is the larger one
    for level, t in zip((0.68, 0.95), thr):
        mass = dens[dens >= t].sum() / dens.sum()
        assert mass == pytest.approx(level, abs=0.01)


def test_scaled_proposal_cov_rule_of_thumb():
    rng = np.random.default_rng(2)
    cov = np.array([[1.0, 0.3], [0.3, 0.5]])
    s = rng.multivariate_normal([0, 0], cov, size=200_000)
    np.testing.assert_allclose(scaled_proposal_cov(s), (2.4 ** 2 / 2) * cov,
                               atol=0.05)
    # the result must be a valid proposal covariance (Cholesky-able)
    GaussianRandomWalk(scaled_proposal_cov(s))


def test_scaled_proposal_cov_1d_and_scale_override():
    s = np.random.default_rng(3).normal(size=(50_000, 1))
    out = scaled_proposal_cov(s, scale=1.0)
    assert out.shape == (1, 1)
    assert out[0, 0] == pytest.approx(1.0, abs=0.03)


def test_scaled_proposal_cov_is_regularized():
    """A rank-deficient warm-up sample (perfectly correlated parameters) still
    gives a proposal covariance that can be Cholesky-factorized (slide 18)."""
    x = np.random.default_rng(4).normal(size=5000)
    s = np.column_stack([x, 2.0 * x])                 # singular sample covariance
    cov = scaled_proposal_cov(s)
    np.testing.assert_allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > 0
    GaussianRandomWalk(cov)                           # would raise if singular


def test_scaled_proposal_cov_rejects_a_chain_that_never_moved():
    with pytest.raises(ValueError, match="no spread"):
        scaled_proposal_cov(np.ones((100, 2)))


def test_dispersed_starts_are_valid_and_reproducible():
    # target is only finite on a disk; the box corners are invalid
    def log_like(theta):
        return 0.0 if theta @ theta < 1.0 else -np.inf

    post = Posterior(log_like, lambda th: 0.0)
    bounds = [(-1.5, 1.5), (-1.5, 1.5)]
    a = dispersed_starts(post, bounds, 6, np.random.default_rng(7))
    b = dispersed_starts(post, bounds, 6, np.random.default_rng(7))
    assert a.shape == (6, 2)
    assert all(np.isfinite(post.log_prob(t)) for t in a)
    np.testing.assert_array_equal(a, b)


def test_dispersed_starts_are_spread_out():
    post = Posterior(lambda th: 0.0, lambda th: 0.0)
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    s = dispersed_starts(post, bounds, 4, np.random.default_rng(3))
    d = [np.linalg.norm(s[i] - s[j]) for i in range(4) for j in range(i)]
    assert min(d) > 0.6            # maximin on the unit square: near the corners


def test_dispersed_starts_raises_when_region_empty():
    post = Posterior(lambda th: -np.inf, lambda th: 0.0)
    with pytest.raises(RuntimeError):
        dispersed_starts(post, [(0, 1)], 2, np.random.default_rng(0),
                         max_tries=50)


def test_dunkley_result_fit_matches_model():
    from cosmo_library.diagnostics import dunkley_model, dunkley_test
    x = np.random.default_rng(4).normal(size=8000)
    res = dunkley_test(x)
    j = np.array([1.0, 10.0, 100.0])
    np.testing.assert_allclose(res.fit(j),
                               dunkley_model(j, res.P0, res.jstar, res.alpha))
    assert dunkley_model(0.0, 2.0, 5.0, 2.0) == pytest.approx(2.0)   # P(0) = P0
