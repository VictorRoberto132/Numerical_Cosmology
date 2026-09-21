"""
Acceptance tests for the convergence diagnostics (milestone 4).

Uses processes with known analytic behaviour:
  * white noise: tau_int = 1, flat spectrum, r ~ 1/N
  * AR(1) x_t = phi x_{t-1} + eps: tau_int = (1+phi)/(1-phi), rho_ell = phi^ell
This lets the checks be quantitative rather than eyeballed.
"""

import numpy as np
import pytest

from cosmo_library.diagnostics import (
    acceptance_rate, autocorr, integrated_time, ess, mcse,
    split_rhat, rank_normalized_rhat, folded_rank_normalized_rhat,
    ess_bulk, ess_quantile, ess_tail, summary,
    power_spectrum, dunkley_test,
)


def ar1(n, phi, rng, sigma=1.0):
    """Stationary AR(1) sample. tau_int = (1+phi)/(1-phi), var = sigma^2/(1-phi^2)."""
    x = np.empty(n)
    x[0] = rng.normal(0.0, sigma / np.sqrt(1.0 - phi ** 2))
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0.0, sigma)
    return x


# --- acceptance rate --------------------------------------------------------
def test_acceptance_rate():
    acc = np.array([True, False, True, True, False])
    assert acceptance_rate(acc) == pytest.approx(0.6)


# --- autocorrelation --------------------------------------------------------
def test_autocorr_iid_is_delta():
    rng = np.random.default_rng(0)
    rho = autocorr(rng.standard_normal(20000))
    assert rho[0] == pytest.approx(1.0)
    # lag-1..20 autocorrelation of IID noise is ~0
    assert np.all(np.abs(rho[1:20]) < 0.05)


def test_autocorr_ar1_matches_phi_power():
    rng = np.random.default_rng(1)
    phi = 0.7
    rho = autocorr(ar1(200000, phi, rng))
    # rho_ell ~ phi^ell for the first few lags
    np.testing.assert_allclose(rho[1:5], phi ** np.arange(1, 5), atol=0.03)


# --- integrated time / ESS --------------------------------------------------
def test_integrated_time_iid():
    rng = np.random.default_rng(2)
    tau = integrated_time(rng.standard_normal(50000))
    assert 0.7 < tau < 1.5                       # ~1 for IID


def test_integrated_time_ar1():
    rng = np.random.default_rng(3)
    phi = 0.8
    tau_true = (1 + phi) / (1 - phi)             # = 9
    tau = integrated_time(ar1(100000, phi, rng))
    assert np.isclose(tau, tau_true, rtol=0.2)


def test_ess_iid_is_near_n():
    rng = np.random.default_rng(4)
    n = 40000
    assert 0.6 * n < ess(rng.standard_normal(n)) < 1.4 * n


def test_ess_multichain_sums():
    rng = np.random.default_rng(5)
    chains = rng.standard_normal((4, 10000))     # 4 IID chains
    # each ~10000 effective -> total ~40000
    assert 0.6 * 40000 < ess(chains) < 1.4 * 40000


def test_mcse_matches_definition():
    rng = np.random.default_rng(6)
    x = rng.standard_normal(30000)
    assert mcse(x) == pytest.approx(np.std(x, ddof=1) / np.sqrt(ess(x)), rel=1e-9)


# --- split-Rhat -------------------------------------------------------------
def test_split_rhat_converged():
    rng = np.random.default_rng(7)
    chains = rng.standard_normal((4, 8000))      # same distribution
    assert split_rhat(chains) < 1.01


def test_split_rhat_detects_different_means():
    rng = np.random.default_rng(8)
    # four chains with clearly different means -> not converged
    chains = np.stack([rng.standard_normal(8000) + 3.0 * k for k in range(4)])
    assert split_rhat(chains) > 1.1


def test_split_rhat_multiparameter_shape():
    rng = np.random.default_rng(9)
    chains = rng.standard_normal((4, 5000, 3))   # 3 parameters
    rhat = split_rhat(chains)
    assert rhat.shape == (3,)
    assert np.all(rhat < 1.01)


def test_rank_normalized_rhat_converged():
    rng = np.random.default_rng(10)
    chains = rng.standard_normal((4, 8000))
    assert rank_normalized_rhat(chains) < 1.01


def test_folded_rhat_converged():
    rng = np.random.default_rng(15)
    chains = rng.standard_normal((4, 8000))
    assert folded_rank_normalized_rhat(chains) < 1.01
    assert folded_rank_normalized_rhat(rng.standard_normal((4, 5000, 3))).shape == (3,)


def test_folded_rhat_detects_scale_mismatch():
    """Same centre, different spread: the location-based Rhats are blind to it
    (slide 20), the folded one is not."""
    rng = np.random.default_rng(16)
    scales = [1.0, 1.0, 5.0, 5.0]
    chains = np.stack([s * rng.standard_normal(8000) for s in scales])
    assert split_rhat(chains) < 1.01
    assert rank_normalized_rhat(chains) < 1.01
    assert folded_rank_normalized_rhat(chains) > 1.1


# --- bulk / tail ESS (slide 21) ---------------------------------------------
def test_ess_bulk_and_tail_iid_are_near_n():
    rng = np.random.default_rng(17)
    n = 40000
    x = rng.standard_normal(n)
    assert 0.6 * n < ess_bulk(x) < 1.4 * n
    assert 0.6 * n < ess_tail(x) < 1.4 * n
    assert 0.6 * n < ess_quantile(x, 0.5) < 1.4 * n


def test_ess_bulk_ar1_matches_theory():
    rng = np.random.default_rng(18)
    phi, n = 0.8, 100000
    tau_true = (1 + phi) / (1 - phi)
    assert np.isclose(ess_bulk(ar1(n, phi, rng)), n / tau_true, rtol=0.3)


def test_ess_multichain_sums_and_tail_is_min_over_quantiles():
    rng = np.random.default_rng(19)
    chains = np.stack([ar1(20000, 0.7, rng) for _ in range(4)])
    probs = (0.05, 0.95)
    assert ess_tail(chains, probs) == pytest.approx(
        min(ess_quantile(chains, p) for p in probs))
    assert ess_bulk(chains) > ess_bulk(chains[0])          # more chains, more ESS


def test_ess_tail_much_smaller_than_bulk_for_slowly_varying_scale():
    """iid sign times a slowly varying scale: the mean is sampled well (bulk-ESS
    ~ N) but the tails are not (tail-ESS collapses), the case slide 21 warns
    about when quoting quantiles and interval endpoints."""
    rng = np.random.default_rng(20)
    n = 40000
    x = np.exp(ar1(n, 0.995, rng, sigma=0.3)) * rng.standard_normal(n)
    assert ess_bulk(x) > 0.9 * n
    assert ess_tail(x) < 0.1 * ess_bulk(x)


def test_summary_reports_folded_rhat_and_bulk_tail_ess():
    rng = np.random.default_rng(21)
    rows = summary(rng.standard_normal((4, 4000, 2)), ["a", "b"])
    for row in rows:
        for key in ("folded_rhat", "ess_bulk", "ess_tail"):
            assert key in row
        assert row["folded_rhat"] < 1.02
        assert row["ess_bulk"] > 0 and row["ess_tail"] > 0


# --- power spectrum / Dunkley ----------------------------------------------
def test_power_spectrum_white_level():
    rng = np.random.default_rng(11)
    x = rng.standard_normal(20000)               # variance ~ 1
    _, P = power_spectrum(x)
    # E[Phat_j] = s^2 ~ 1 for white noise; the mean is a stable estimator
    assert 0.8 < P.mean() < 1.2


def test_dunkley_iid_converges():
    rng = np.random.default_rng(12)
    n = 50000
    dk = dunkley_test(rng.standard_normal(n))
    assert dk.jstar_ok                           # no turnover -> j* large
    assert dk.r < 0.01                           # r ~ 1/N is tiny
    assert dk.converged


def test_dunkley_r_matches_inverse_ess():
    """The spectral r and the autocorrelation ESS estimate the same thing:
    r = P0/(N s^2) should match 1/ESS within the fit noise (slide 23)."""
    rng = np.random.default_rng(13)
    x = ar1(100000, 0.8, rng)                    # tau_true = 9
    dk = dunkley_test(x)
    r_from_ess = 1.0 / ess(x)
    assert 0.3 < dk.r / r_from_ess < 3.0         # consistent within fit scatter
    assert dk.jstar < np.inf                     # a real turnover exists
    assert 0.5 < dk.alpha < 4.0                  # random-walk-like slope


def test_dunkley_correlated_worse_than_iid():
    rng = np.random.default_rng(14)
    n = 40000
    r_iid = dunkley_test(rng.standard_normal(n)).r
    r_cor = dunkley_test(ar1(n, 0.9, rng)).r
    assert r_cor > r_iid                          # correlation inflates Var(xbar)