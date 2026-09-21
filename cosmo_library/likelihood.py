"""
Union2.1 posterior in (Omega_m, Omega_lambda), no flatness imposed (milestone 3).

Wires the FLRW background (mu_shape) to the marginalized-M likelihood and a
box prior, producing a bayes.Posterior the sampler can consume unchanged.
E^2(z) < 0 or non-finite distances propagate to log_prob = -inf, which the
sampler treats as a rejection (slides 15, 32).
"""

import numpy as np

from .bayes import Posterior
from .cosmology import FLRW


def make_box_log_prior(bounds):
    """Uniform log-prior on a hyper-rectangle.

    bounds : sequence of (low, high) per parameter, e.g.
             [(0, 2), (-1, 3)] for the slide-32 example prior on
             (Omega_m, Omega_lambda). Returns -inf outside the box.
    """
    lows = np.array([b[0] for b in bounds], float)
    highs = np.array([b[1] for b in bounds], float)

    def log_prior(theta):
        theta = np.asarray(theta, float)
        if np.any(theta < lows) or np.any(theta > highs):
            return -np.inf
        return 0.0
    return log_prior


def make_union21_posterior(data, H0=70.0,
                           bounds=((0.0, 2.0), (-1.0, 3.0))):
    """
    Build the (Omega_m, Omega_lambda) posterior for the Union2.1 fit.

    Parameters
    ----------
    data : Union21
        The loaded dataset with .z and .log_likelihood.
    H0 : float
        Fiducial Hubble constant. Its value is degenerate with the marginalized
        zero point M, so the result is insensitive to it (any positive value).
    bounds : ((om_lo, om_hi), (ol_lo, ol_hi))
        Box-prior limits; default is the slide-32 example prior.

    Returns
    -------
    Posterior
        theta = (Omega_m, Omega_lambda) -> log posterior.
    """
    z = data.z

    def model(theta):                       # (om, ol) -> mu_shape(z)
        om, ol = float(theta[0]), float(theta[1])
        cosmo = FLRW(H0, om, ol)
        return cosmo.distance_modulus(z)    # NaN where E^2 <= 0

    def log_likelihood(mu_theory):
        return data.log_likelihood(mu_theory)

    return Posterior(log_likelihood, make_box_log_prior(bounds), model=model)