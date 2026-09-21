"""Union2.1 data loader and marginalized-M likelihood (milestone 3).

The supernova file is a LaTeX-style table: fields separated by '&', and each
record (one SN) ends with a backslash. Records wrap over 2 physical lines, or
1 for the rejected SNe with no photometry, so line-based tools misalign; we
split on the backslash terminator instead. The covariance is the
_sys matrix: its diagonal ALREADY contains the statistical variances, so the
per-SN mu errors in the table are parsed but never re-added (slide 32).

The nuisance zero point M (slide 33) is marginalized analytically with a flat
prior. Since M enters linearly in a Gaussian chi^2,
    chi2(M) = A - 2 B M + E M^2,   A = d^T C^-1 d, B = 1^T C^-1 d, E = 1^T C^-1 1,
completing the square and integrating gives
    chi2_marg = A - B^2/E   (+ const),   log L = -1/2 (A - B^2/E).
All C^-1 products use a Cholesky factorization; C^-1 is never formed.
"""

import os
import re

import numpy as np
from scipy.linalg import cho_factor, cho_solve

_VAL_ERR = re.compile(r"^([-+0-9.eE]+)\(([-+0-9.eE]+)\)$")
# A record is: name, z, m_B, x1, color, mu(err), P_low, sample_ID, cut_code.
# cut_code is empty for the SNe that pass the release cuts and holds letter
# code(s) ('z', 'p', 'f', 'p,z', ...) for the rejected ones.
_N_FIELDS = 9
_IZ, _IMU, _ICUT = 1, 5, 8
_CUT_CODE = re.compile(r"^([a-z](,[a-z])*)?$")


def _parse_sne_table(path):
    """Return (names, z, mu, flag) from the Union2.1 AllSNe LaTeX table.

    Records are delimited by the trailing backslash, not by line breaks or a
    fixed field count. `flag` is 1 for SNe that pass the cuts (empty cut code)
    and 0 for rejected ones. `mu` is the value of 'mu(err)' only; the
    statistical error is dropped because the _sys covariance already has it.
    Rejected SNe without a fit have no mu, which is returned as NaN.
    """
    with open(path) as f:
        raw = f.read()

    chunks = raw.split("\\")
    if chunks.pop().strip():                           # text after the last '\'
        raise ValueError("unterminated record after the last backslash")

    names, z, mu, flag = [], [], [], []
    for chunk in chunks:
        fields = [tok.strip() for tok in chunk.split("&")]   # \n is just wrap
        if len(fields) != _N_FIELDS:
            raise ValueError(
                f"record {fields[0]!r} has {len(fields)} fields, expected "
                f"{_N_FIELDS}; the table format may differ from Union2.1"
            )
        if not _CUT_CODE.match(fields[_ICUT]):
            raise ValueError(
                f"unexpected cut code {fields[_ICUT]!r} in record {fields[0]}")

        if fields[_IMU] == "":                         # rejected, no light-curve fit
            mu_val = np.nan
        else:
            m = _VAL_ERR.match(fields[_IMU])           # 'mu(err)' -> value only
            if m is None:
                raise ValueError(
                    f"cannot parse mu field {fields[_IMU]!r} in record {fields[0]}")
            mu_val = float(m.group(1))                 # statistical err discarded

        names.append(fields[0])
        z.append(float(fields[_IZ]))
        mu.append(mu_val)
        flag.append(int(fields[_ICUT] == ""))
    return (np.array(names), np.array(z, float),
            np.array(mu, float), np.array(flag, int))


def _load_covariance(path, n_expected):
    """Load the _sys covariance as an (n, n) matrix, tolerant of line breaks."""
    flat = np.loadtxt(path).ravel()                    # ignores newlines
    n2 = flat.size
    n = int(round(np.sqrt(n2)))
    if n * n != n2:
        raise ValueError(f"covariance has {n2} numbers, not a perfect square")
    if n != n_expected:
        raise ValueError(
            f"covariance is {n}x{n} but {n_expected} SNe passed the cut"
        )
    return flat.reshape(n, n)


class Union21:
    """Union2.1 dataset with a marginalized-M Gaussian likelihood.

    Parameters
    ----------
    data_dir : str
        Directory holding SCPUnion2.1_AllSNe.txt and
        SCPUnion2.1_covmat_sys.txt.

    Attributes
    ----------
    z, mu : ndarray
        Redshift and observed distance modulus of the selected SNe.
    cov : ndarray
        The (n, n) systematics covariance (statistical errors included).
    n : int
        Number of selected SNe (580 for the standard release).
    """

    SNE_FILE = "SCPUnion2.1_AllSNe.txt"
    COV_FILE = "SCPUnion2.1_covmat_sys.txt"

    def __init__(self, data_dir):
        names, z_all, mu_all, flag = _parse_sne_table(
            os.path.join(data_dir, self.SNE_FILE))
        sel = flag == 1                                # release selection cut
        self.names = names[sel]
        self.z = z_all[sel]
        self.mu = mu_all[sel]
        self.n = int(sel.sum())

        self.cov = _load_covariance(
            os.path.join(data_dir, self.COV_FILE), self.n)

        # factor once; reused every likelihood call (C is fixed)
        self._cho = cho_factor(self.cov, lower=True)
        self._ones = np.ones(self.n)
        self._Cinv_ones = cho_solve(self._cho, self._ones)
        self._E = float(self._ones @ self._Cinv_ones)  # 1^T C^-1 1, constant
        # constant part of the marginalized log-likelihood: 1/2 log(2 pi / E)
        self._log_norm = 0.5 * np.log(2.0 * np.pi / self._E)

    def log_likelihood(self, mu_theory):
        """Marginalized-M Gaussian log-likelihood for a model's mu(z).

        `mu_theory` is mu_shape(z; Omega_m, Omega_lambda) at the dataset's
        redshifts (the additive M is integrated out here, so do NOT add it).
        Returns -inf if the model produced any non-finite mu (invalid E^2).
        """
        mu_theory = np.asarray(mu_theory, dtype=float)
        if mu_theory.shape != self.z.shape or not np.all(np.isfinite(mu_theory)):
            return -np.inf                             # invalid cosmology

        d = self.mu - mu_theory                        # residual, M not applied
        Cinv_d = cho_solve(self._cho, d)
        A = float(d @ Cinv_d)
        B = float(self._ones @ Cinv_d)
        return -0.5 * (A - B * B / self._E) + self._log_norm

    def best_fit_M(self, mu_theory):
        """The profiled zero point M_hat = B/E for a given model (diagnostic)."""
        d = self.mu - np.asarray(mu_theory, dtype=float)
        B = float(self._ones @ cho_solve(self._cho, d))
        return B / self._E