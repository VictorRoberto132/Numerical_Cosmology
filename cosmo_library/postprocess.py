"""
Posterior post-processing: credible intervals and 2-D contour levels.

Numerical helpers only (no plotting), so the library keeps numpy/scipy as its
sole dependencies; the notebook draws with the arrays returned here.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def credible_interval(x, level=0.68):
    """Median and equal-tailed credible interval of a 1-D sample.

    Returns (median, lower, upper) where [lower, upper] holds `level` of the
    posterior mass with (1 - level)/2 in each tail (16th-84th percentiles for
    the default 68%).
    """
    x = np.asarray(x, dtype=float).ravel()
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    tail = 50.0 * (1.0 - level)
    lo, med, hi = np.percentile(x, [tail, 50.0, 100.0 - tail])
    return float(med), float(lo), float(hi)


def contour_levels_2d(x, y, levels=(0.68, 0.95), bins=80, smooth=1.5):
    """Density grid and highest-density-region thresholds for 2-D samples.

    The samples are histogrammed and Gaussian-smoothed (`smooth`, in bins).
    For each mass fraction in `levels`, the returned threshold is the density
    such that the region {density >= threshold} encloses that fraction of the
    posterior mass (a highest-posterior-density contour).

    Returns
    -------
    xc, yc : 1-D arrays of bin centers
    density : (len(yc), len(xc)) smoothed histogram, ready for plt.contour
    thresholds : array aligned with `levels`. Sort ascending before passing
        to plt.contour, since the 95% threshold is lower than the 68% one.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    hist, xe, ye = np.histogram2d(x, y, bins=bins)
    hist = gaussian_filter(hist, smooth)

    order = np.sort(hist.ravel())[::-1]
    cum = np.cumsum(order) / order.sum()
    thresholds = np.array([order[np.searchsorted(cum, lv)] for lv in levels])

    xc = 0.5 * (xe[1:] + xe[:-1])
    yc = 0.5 * (ye[1:] + ye[:-1])
    return xc, yc, hist.T, thresholds
