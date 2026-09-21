"""Acceptance tests for the Union2.1 loader and likelihood (milestone 3).

Required checks: data/covariance dimensions agree, Cholesky solves are used,
and the marginalized-M likelihood is invariant to a global mu shift (the
defining property of integrating out the zero point).

Set UNION_DATA_DIR to the folder with the two SCPUnion2.1 files; if unset,
the data-dependent tests are skipped and only the pure-math likelihood test
(on a synthetic covariance) runs.
"""

import os

import numpy as np
import pytest

from cosmo_library.data import Union21, _parse_sne_table, _load_covariance

DATA_DIR = os.environ.get(
    "UNION_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "..", "union_data"),
)
_has_data = os.path.isdir(DATA_DIR) and os.path.isfile(
    os.path.join(DATA_DIR, Union21.SNE_FILE))
needs_data = pytest.mark.skipif(not _has_data, reason="Union2.1 files not found")


# --- data-dependent acceptance tests ---------------------------------------
@needs_data
def test_selection_gives_580():
    data = Union21(DATA_DIR)
    assert data.n == 580                          # the release cut


@needs_data
def test_dimensions_agree():
    data = Union21(DATA_DIR)
    assert data.z.shape == (580,)
    assert data.mu.shape == (580,)
    assert data.cov.shape == (580, 580)


@needs_data
def test_covariance_is_spd():
    data = Union21(DATA_DIR)
    # symmetric...
    np.testing.assert_allclose(data.cov, data.cov.T, rtol=0, atol=1e-10)
    # ...and positive-definite (Cholesky succeeds, all eigenvalues > 0)
    assert np.all(np.linalg.eigvalsh(data.cov) > 0)


@needs_data
def test_shift_invariance_on_real_data():
    data = Union21(DATA_DIR)
    mu_model = data.mu.copy()                     # perfect fit -> residual 0
    base = data.log_likelihood(mu_model)
    shifted = data.log_likelihood(mu_model - 5.0) # global -5 mag offset
    assert np.isclose(base, shifted)              # M absorbs it


@needs_data
def test_invalid_model_gives_minus_inf():
    data = Union21(DATA_DIR)
    bad = np.full(data.n, np.nan)
    assert data.log_likelihood(bad) == -np.inf


# --- pure-math test on a synthetic covariance (always runs) ----------------
def _synthetic_dataset(n=40, seed=0):
    """A Union21-like object with a random SPD covariance, no files needed."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    cov = A @ A.T + n * np.eye(n)

    obj = Union21.__new__(Union21)                # bypass file loading
    from scipy.linalg import cho_factor, cho_solve
    obj.z = np.linspace(0.01, 1.5, n)
    obj.mu = rng.standard_normal(n) + 40.0
    obj.cov = cov
    obj.n = n
    obj._cho = cho_factor(cov, lower=True)
    obj._ones = np.ones(n)
    obj._Cinv_ones = cho_solve(obj._cho, obj._ones)
    obj._E = float(obj._ones @ obj._Cinv_ones)
    obj._log_norm = 0.5 * np.log(2.0 * np.pi / obj._E)
    return obj


def test_marginalization_matches_profile():
    """log L_marg uses A - B^2/E; the profiled chi2_min must equal A - B^2/E."""
    from scipy.linalg import cho_solve
    data = _synthetic_dataset()
    mu_model = data.mu - 0.3          # arbitrary residual
    d = data.mu - mu_model
    Cinv_d = cho_solve(data._cho, d)
    A = d @ Cinv_d
    B = data._ones @ Cinv_d
    chi2_marg = A - B * B / data._E

    M_hat = data.best_fit_M(mu_model) # = B/E
    dd = d - M_hat * data._ones
    chi2_profile = dd @ cho_solve(data._cho, dd)

    assert np.isclose(chi2_marg, chi2_profile)
    assert np.isclose(M_hat, B / data._E)


def test_shift_invariance_synthetic():
    data = _synthetic_dataset()
    mu_model = data.mu - 0.2
    base = data.log_likelihood(mu_model)
    for shift in (1.0, -3.5, 50.0):
        assert np.isclose(data.log_likelihood(mu_model - shift), base)