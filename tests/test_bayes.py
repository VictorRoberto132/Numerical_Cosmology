"""Acceptance tests for the Bayesian core (slide 35, milestone 2).

Three required checks:
  1. recover a known Gaussian target (mean and covariance);
  2. reproduce a run from a seed (and differ under a different seed);
  3. rejected states are repeated in the stored chain.

Plus two robustness checks (slide 13):
  - an invalid initial state is rejected up front;
  - a proposal into a zero-prior region is rejected, not crashed on.
"""

import numpy as np
import pytest

from cosmo_library.bayes import (
    Posterior, GaussianRandomWalk, MetropolisHastings,
)


def make_gaussian_posterior(mu, cov):
    """A Posterior whose density is exactly N(mu, cov) (flat 'prior')."""
    mu = np.asarray(mu, dtype=float)
    inv = np.linalg.inv(np.asarray(cov, dtype=float))

    def log_likelihood(theta):
        d = theta - mu
        return -0.5 * d @ inv @ d

    def log_prior(theta):
        return 0.0                      # improper flat; fine as a test target

    return Posterior(log_likelihood, log_prior)


# --- 1. recover a known Gaussian target ------------------------------------
def test_recovers_gaussian_target():
    mu_true = np.array([0.5, -1.0])
    cov_true = np.array([[1.0, 0.3],
                         [0.3, 0.5]])
    post = make_gaussian_posterior(mu_true, cov_true)

    # rule-of-thumb proposal scale for a d-dim Gaussian: 2.4^2/d * target cov
    prop = GaussianRandomWalk((2.4 ** 2 / 2) * cov_true)
    mh = MetropolisHastings(post, prop)

    rng = np.random.default_rng(0)
    res = mh.sample(np.zeros(2), n_steps=60000, rng=rng)

    chain = res.samples[10000:]                 # drop warm-up
    mean_hat = chain.mean(axis=0)
    cov_hat = np.cov(chain, rowvar=False)

    np.testing.assert_allclose(mean_hat, mu_true, atol=0.15)
    np.testing.assert_allclose(cov_hat, cov_true, atol=0.15)
    assert 0.15 < res.acceptance_rate < 0.6     # sane mixing regime


# --- 2. reproducibility from a seed ----------------------------------------
def test_seed_reproducibility():
    post = make_gaussian_posterior([0.0, 0.0], np.eye(2))
    mh = MetropolisHastings(post, GaussianRandomWalk(0.5 * np.eye(2)))
    x0 = np.zeros(2)

    r1 = mh.sample(x0, 500, rng=np.random.default_rng(42))
    r2 = mh.sample(x0, 500, rng=np.random.default_rng(42))
    r3 = mh.sample(x0, 500, rng=np.random.default_rng(7))

    assert np.array_equal(r1.samples, r2.samples)       # same seed -> identical
    assert np.array_equal(r1.accepted, r2.accepted)
    assert not np.array_equal(r1.samples, r3.samples)   # different seed -> differs


# --- 3. rejected states are repeated ---------------------------------------
def test_rejected_states_are_repeated():
    post = make_gaussian_posterior([0.0, 0.0], np.eye(2))
    # deliberately oversized proposal -> plenty of rejections
    mh = MetropolisHastings(post, GaussianRandomWalk(25.0 * np.eye(2)))

    res = mh.sample(np.zeros(2), 2000, rng=np.random.default_rng(1))

    n_rej = (~res.accepted).sum()
    assert 0 < n_rej < 2000                     # some accepted, some rejected

    for t in range(1, 2000):
        if not res.accepted[t]:
            assert np.array_equal(res.samples[t], res.samples[t - 1])
            assert res.log_probs[t] == res.log_probs[t - 1]

    assert np.isclose(res.acceptance_rate, res.accepted.mean())


# --- robustness: invalid initial state -------------------------------------
def test_invalid_initial_state_raises():
    def log_likelihood(theta):
        return 0.0

    def log_prior(theta):                       # support is the unit box
        return 0.0 if np.all(np.abs(theta) < 1.0) else -np.inf

    post = Posterior(log_likelihood, log_prior)
    mh = MetropolisHastings(post, GaussianRandomWalk(0.1 * np.eye(2)))

    with pytest.raises(ValueError):
        mh.sample(np.array([5.0, 5.0]), 100, rng=np.random.default_rng(0))


# --- robustness: -inf target is a rejection, not a crash -------------------
def test_out_of_prior_proposals_are_rejected():
    def log_likelihood(theta):
        return 0.0

    def log_prior(theta):
        return 0.0 if np.all(np.abs(theta) < 1.0) else -np.inf

    post = Posterior(log_likelihood, log_prior)
    # big proposal frequently lands outside the box -> those must be rejected
    mh = MetropolisHastings(post, GaussianRandomWalk(4.0 * np.eye(2)))

    res = mh.sample(np.zeros(2), 3000, rng=np.random.default_rng(2))

    # every stored state stays inside the support and has finite log-prob
    assert np.all(np.abs(res.samples) < 1.0)
    assert np.all(np.isfinite(res.log_probs))
    assert (~res.accepted).sum() > 0            # rejections did happen

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"])) 