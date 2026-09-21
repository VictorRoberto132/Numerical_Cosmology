"""
# In this file, we define the cosmology module, with FLRW cosmology (H0, omega_m, omega_lambda, etc)

All distances are returned in Mpc, times in Gyr. The class is deliberately
thin: it only knows the background expansion. Likelihood, priors, proposals
and diagnostics live in other modules.
"""

import numpy as np
from scipy.integrate import quad

try:                                   # scipy >= 1.6
    from scipy.integrate import cumulative_trapezoid
except ImportError:                    # older scipy
    from scipy.integrate import cumtrapz as cumulative_trapezoid

# --- physical constants -----------------------------------------------------
C_KM_S      = 299792.458                 # speed of light [km/s]
KM_PER_MPC  = 3.0856775814913673e19      # km per Mpc
SEC_PER_GYR = 3.1556952e16               # seconds per Gyr


class FLRW:
    """
    Flat-or-curved FLRW model with matter, radiation and a cosmological
    constant.

    Parameters
    ----------
    H0 : float
        Hubble constant [km/s/Mpc].
    omega_m, omega_lambda : float
        Present-day matter and dark-energy density parameters.
    omega_r : float, optional
        Present-day radiation density parameter (default 0).

    Notes
    -----
    Curvature is derived: Omega_k = 1 - Omega_m - Omega_lambda - Omega_r,
    so E(0) = 1 always holds. Methods return NaN for cosmologies whose
    E^2(z) becomes non-positive on the requested range: that is the signal
    the likelihood uses to reject the parameter point.
    """

    def __init__(self, H0, omega_m, omega_lambda, omega_r=0.0):
        self.H0 = float(H0)
        self.omega_m = float(omega_m)
        self.omega_lambda = float(omega_lambda)
        self.omega_r = float(omega_r)
        self.omega_k = 1.0 - self.omega_m - self.omega_lambda - self.omega_r

        self.hubble_distance = C_KM_S / self.H0                       # Mpc
        self.hubble_time_gyr = (KM_PER_MPC / self.H0) / SEC_PER_GYR   # Gyr

    def __repr__(self):
        return (f"FLRW(H0={self.H0}, omega_m={self.omega_m}, "
                f"omega_lambda={self.omega_lambda}, omega_r={self.omega_r}, "
                f"omega_k={self.omega_k:.4f})")

    @property
    def is_flat(self):
        return abs(self.omega_k) < 1e-8

    # --- expansion rate -----------------------------------------------------
    def E2(self, z):
        """Dimensionless (H/H0)^2. May be non-positive for unphysical models."""
        z = np.asarray(z, dtype=float)
        zp1 = 1.0 + z
        return (self.omega_r * zp1**4
                + self.omega_m * zp1**3
                + self.omega_k * zp1**2
                + self.omega_lambda)

    def E(self, z):
        """Dimensionless expansion rate H(z)/H0; NaN where E^2 <= 0."""
        e2 = self.E2(z)
        with np.errstate(invalid="ignore"):
            return np.where(e2 > 0.0, np.sqrt(e2), np.nan)

    def H(self, z):
        """Hubble parameter [km/s/Mpc]."""
        return self.H0 * self.E(z)

    # --- comoving / metric distances ---------------------------------------
    def comoving_distance(self, z, n_grid=2048):
        """Line-of-sight comoving distance chi(z) [Mpc].

        chi = D_H * int_0^z dz'/E(z'), evaluated with a single cumulative
        trapezoid pass so that array inputs cost one integration, not one
        per redshift (this is what makes it usable inside the MCMC loop).
        Returns NaN if E^2 <= 0 anywhere on [0, max(z)].
        """
        z = np.asarray(z, dtype=float)
        scalar = z.ndim == 0
        z = np.atleast_1d(z)
        if np.any(z < 0):
            raise ValueError("redshift must be non-negative")

        zmax = float(z.max())
        if zmax == 0.0:
            chi = np.zeros_like(z)
            return chi[0] if scalar else chi

        grid = np.linspace(0.0, zmax, n_grid)
        e2 = self.E2(grid)
        if np.any(e2 <= 0.0):                    # invalid model -> reject
            chi = np.full_like(z, np.nan)
            return chi[0] if scalar else chi

        integrand = 1.0 / np.sqrt(e2)
        cumint = cumulative_trapezoid(integrand, grid, initial=0.0)
        chi = self.hubble_distance * np.interp(z, grid, cumint)
        return chi[0] if scalar else chi

    def comoving_transverse_distance(self, z, **kw):
        """Transverse comoving distance D_M(z) [Mpc], curvature-aware."""
        chi = self.comoving_distance(z, **kw)
        DH = self.hubble_distance
        if self.is_flat:
            return chi
        sqrtk = np.sqrt(abs(self.omega_k))
        if self.omega_k > 0.0:                   # open universe
            return DH / sqrtk * np.sinh(sqrtk * chi / DH)
        return DH / sqrtk * np.sin(sqrtk * chi / DH)   # closed universe

    def angular_diameter_distance(self, z, **kw):
        """Angular diameter distance D_A(z) [Mpc]."""
        z = np.asarray(z, dtype=float)
        return self.comoving_transverse_distance(z, **kw) / (1.0 + z)

    def luminosity_distance(self, z, **kw):
        """Luminosity distance D_L(z) [Mpc]."""
        z = np.asarray(z, dtype=float)
        return (1.0 + z) * self.comoving_transverse_distance(z, **kw)

    def distance_modulus(self, z, **kw):
        """Distance modulus mu(z) = 5 log10(D_L/Mpc) + 25.

        Carries the H0 zero point through D_H = c/H0. In the SNe fit this
        additive constant is degenerate with the nuisance M:
        use mu_th = distance_modulus(z) + M with a fixed fiducial H0.
        """
        DL = self.luminosity_distance(z, **kw)   # Mpc
        with np.errstate(invalid="ignore", divide="ignore"):
            return 5.0 * np.log10(DL) + 25.0

    # --- times --------------------------------------------------------------
    def _time_integral(self, z_lo, z_hi):
        def integrand(zp):
            e2 = self.E2(zp)
            if e2 <= 0.0:
                return np.nan
            return 1.0 / ((1.0 + zp) * np.sqrt(e2))
        val, _ = quad(integrand, z_lo, z_hi)
        return self.hubble_time_gyr * val

    def age(self, z=0.0):
        """Age of the universe at redshift z [Gyr]: t = t_H int_z^inf dz'/[(1+z')E]."""
        return self._time_integral(float(z), np.inf)

    def lookback_time(self, z):
        """Lookback time to redshift z [Gyr]: t_H int_0^z dz'/[(1+z')E]."""
        return self._time_integral(0.0, float(z))

    def conformal_time(self, z=0.0):
        """Conformal time at z [Gyr], eta = t_H int_z^inf dz'/E(z')."""
        def integrand(zp):
            e2 = self.E2(zp)
            if e2 <= 0.0:
                return np.nan
            return 1.0 / np.sqrt(e2)
        val, _ = quad(integrand, float(z), np.inf)
        return self.hubble_time_gyr * val

    # --- volume -------------------------------------------------------------
    def comoving_volume_element(self, z, **kw):
        """Comoving volume element dV_c/(dz dOmega) [Mpc^3/sr]."""
        DM = self.comoving_transverse_distance(z, **kw)
        return self.hubble_distance * DM**2 / self.E(z)