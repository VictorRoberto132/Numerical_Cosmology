"""Acceptance tests for the FLRW background.

Here, we do four tests to check that the cosmology class is implemented correctly:
  1. E(0) = 1
  2. low-z Hubble law: D_L ~ c z / H0
  3. D_M = chi in the flat case
  4. D_L = (1+z)^2 D_A 

Plus two robustness checks the Bayesian core relies on:
  - invalid E^2(z) -> NaN (the rejection signal used by the likelihood)
  - curvature branch is actually applied (open > chi > closed)
"""

import numpy as np
import pytest

from cosmo_library.cosmology import FLRW, C_KM_S


# A few valid models spanning flat / open / closed geometries.
FLAT   = FLRW(70.0, 0.3, 0.7)          # Omega_k = 0
OPEN   = FLRW(70.0, 0.3, 0.0)          # Omega_k = +0.7
CLOSED = FLRW(70.0, 0.3, 1.2)          # Omega_k = -0.5


# --- 1. E(0) = 1 by construction -------------------------------------------
@pytest.mark.parametrize("cosmo", [FLAT, OPEN, CLOSED])
def test_E_at_z0_is_one(cosmo):
    # E^2(0) = Omega_m + Omega_k + Omega_lambda + Omega_r = 1 always.
    assert np.isclose(cosmo.E(0.0), 1.0, rtol=1e-12, atol=1e-12)


# --- 2. low-z Hubble law: D_L -> c z / H0 ----------------------------------
def test_low_z_hubble_law():
    z = 1e-4
    expected = C_KM_S * z / FLAT.H0            # c z / H0  [Mpc]
    assert np.isclose(FLAT.luminosity_distance(z), expected, rtol=1e-3)


# --- 3. D_M = chi in the flat case -----------------------------------------
def test_transverse_equals_comoving_when_flat():
    z = np.array([0.01, 0.1, 0.5, 1.0, 1.5])
    assert FLAT.is_flat
    np.testing.assert_allclose(
        FLAT.comoving_transverse_distance(z),
        FLAT.comoving_distance(z),
        rtol=1e-12, atol=1e-9,
    )


# --- 4. Etherington reciprocity: D_L = (1+z)^2 D_A --------------------------
@pytest.mark.parametrize("cosmo", [FLAT, OPEN, CLOSED])
def test_etherington_reciprocity(cosmo):
    z = np.array([0.05, 0.2, 0.7, 1.3])
    DL = cosmo.luminosity_distance(z)
    DA = cosmo.angular_diameter_distance(z)
    np.testing.assert_allclose(DL, (1.0 + z) ** 2 * DA, rtol=1e-12, atol=1e-9)


# --- robustness: invalid E^2(z) must produce NaN, not an exception ---------
def test_invalid_cosmology_returns_nan():
    # Omega_m=0.3, Omega_lambda=3.0 -> Omega_k=-2.3; E^2(z) dips below 0
    # around z~0.5, so distances on [0, 1] are undefined.
    bad = FLRW(70.0, 0.3, 3.0)
    assert np.isnan(bad.comoving_distance(1.0))
    assert np.isnan(bad.distance_modulus(1.0))
    # The valid models must NOT be NaN on the same range.
    assert np.isfinite(FLAT.distance_modulus(1.0))


# --- robustness: curvature branch is really applied ------------------------
def test_curvature_bends_distances():
    z = 1.0
    chi_open   = OPEN.comoving_distance(z)
    chi_closed = CLOSED.comoving_distance(z)
    # sinh(x) > x  (open) ;  sin(x) < x  (closed)
    assert OPEN.comoving_transverse_distance(z)   > chi_open
    assert CLOSED.comoving_transverse_distance(z) < chi_closed


# --- rest of the slide-34 interface, against Einstein-de Sitter analytics ---
# EdS (Omega_m = 1, Omega_lambda = 0): E(z) = (1+z)^{3/2}, so every quantity
# below has a closed form.
EDS = FLRW(70.0, 1.0, 0.0)


def test_hubble_parameter_eds():
    z = np.array([0.0, 0.5, 1.0, 2.0])
    np.testing.assert_allclose(EDS.H(z), 70.0 * (1.0 + z) ** 1.5, rtol=1e-12)


def test_times_eds():
    tH = EDS.hubble_time_gyr
    assert EDS.age(0.0) == pytest.approx(2.0 / 3.0 * tH, rel=1e-8)
    assert EDS.age(1.0) == pytest.approx(2.0 / 3.0 * tH * 2.0 ** -1.5, rel=1e-8)
    assert EDS.lookback_time(1.0) == pytest.approx(
        2.0 / 3.0 * tH * (1.0 - 2.0 ** -1.5), rel=1e-8)
    # age(z) + lookback(z) = age(0)
    assert EDS.age(1.0) + EDS.lookback_time(1.0) == pytest.approx(EDS.age(0.0), rel=1e-8)
    assert EDS.conformal_time(0.0) == pytest.approx(2.0 * tH, rel=1e-8)


def test_comoving_volume_element_eds():
    z = np.array([0.3, 1.0])
    DH = EDS.hubble_distance
    DM = 2.0 * DH * (1.0 - 1.0 / np.sqrt(1.0 + z))
    np.testing.assert_allclose(EDS.comoving_transverse_distance(z), DM, rtol=1e-6)
    np.testing.assert_allclose(EDS.comoving_volume_element(z),
                               DH * DM ** 2 / (1.0 + z) ** 1.5, rtol=1e-6)

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))