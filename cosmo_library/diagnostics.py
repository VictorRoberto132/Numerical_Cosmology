"""
Traces, ACF/ESS, split-R̂, espectro (P0, j*, α, r)

Convergence diagnostics (slide 34 architecture; milestone 4).

Everything operates on unthinned, post-warm-up chains (slide 24). Single-
parameter functions take a 1-D array; multi-chain functions take
(n_chains, n_steps). The Union2.1 notebook loops these over both parameters.

Provides:
  * acceptance_rate
  * autocorr, integrated_time, ess, mcse       (slide 21)
  * ess_bulk, ess_quantile, ess_tail           (slide 21)
  * split_rhat, rank_normalized_rhat,
    folded_rank_normalized_rhat                (slide 20)
  * power_spectrum, dunkley_test               (slides 22-24)
  * summary                                    (report table, milestone 4)
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm, rankdata

EULER_GAMMA = 0.5772156649015329


# ===========================================================================
#  Acceptance
# ===========================================================================
def acceptance_rate(accepted):
    """Fraction of accepted proposals. `accepted` is a boolean array."""
    return float(np.mean(np.asarray(accepted, dtype=bool)))


# ===========================================================================
#  Autocorrelation, integrated time, ESS, MCSE  (slide 21)
# ===========================================================================
def _next_pow_two(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def autocorr(x):
    """Normalized autocorrelation rho_ell of a 1-D chain, via FFT.

    Returns an array with rho[0] = 1. A constant chain yields rho = [1, 0, ...].
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    xc = x - x.mean()
    var = np.dot(xc, xc)
    if var == 0.0:                          # constant chain
        rho = np.zeros(n)
        rho[0] = 1.0
        return rho

    nfft = 2 * _next_pow_two(n)
    f = np.fft.fft(xc, n=nfft)
    acf = np.fft.ifft(f * np.conj(f))[:n].real
    acf /= acf[0]
    return acf


def _auto_window(tau_cum, c=5.0):
    """Sokal's automatic windowing: smallest M with M >= c * tau(M)."""
    below = np.arange(tau_cum.size) < c * tau_cum
    if np.all(below):                       # chain too short to satisfy it
        return tau_cum.size - 1
    return int(np.argmin(below))            # first index where it fails


def integrated_time(x, c=5.0):
    """Integrated autocorrelation time tau_int = 1 + 2 sum_ell rho_ell,
    with Sokal's stable truncation. Never returns less than 1.
    """
    rho = autocorr(x)
    tau_cum = 2.0 * np.cumsum(rho) - 1.0    # tau(M) = 1 + 2 sum_{1}^{M} rho
    window = _auto_window(tau_cum, c)
    return float(max(tau_cum[window], 1.0))


def ess(x, c=5.0):
    """Effective sample size N/tau_int.

    1-D input -> single chain. (n_chains, n_steps) input -> sum of per-chain
    ESS (valid for independent, stationary chains; conservative otherwise).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return float(x.size / integrated_time(x, c))
    if x.ndim == 2:
        return float(sum(x.shape[1] / integrated_time(chain, c) for chain in x))
    raise ValueError("ess expects a 1-D or 2-D (n_chains, n_steps) array")


def mcse(x, c=5.0):
    """Monte Carlo standard error of the posterior mean, s / sqrt(ESS)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        s = np.std(x, ddof=1)
        return float(s / np.sqrt(ess(x, c)))
    if x.ndim == 2:
        s = np.std(x.ravel(), ddof=1)       # pooled marginal spread
        return float(s / np.sqrt(ess(x, c)))
    raise ValueError("mcse expects a 1-D or 2-D array")


def _as_chains_2d(x):
    """View a 1-D chain as (1, n); pass an (n_chains, n_steps) array through."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x[None, :]
    if x.ndim == 2:
        return x
    raise ValueError("expected a 1-D or 2-D (n_chains, n_steps) array")


def ess_bulk(x, c=5.0):
    """Bulk-ESS (slide 21): ESS of the rank-normalized draws.

    The appropriate ESS for central summaries (mean, median). Same input
    shapes as `ess` (1-D chain, or (n_chains, n_steps) -> sum over chains).
    """
    return ess(_rank_normalize(_as_chains_2d(x)), c)


def ess_quantile(x, prob, c=5.0):
    """ESS for the `prob`-quantile: the ESS of the indicator 1[x <= q_prob].

    The precision of a quantile is set by how fast the chain moves across
    that quantile, not across the bulk.
    """
    chains = _as_chains_2d(x)
    q = np.quantile(chains, prob)
    return ess((chains <= q).astype(float), c)


def ess_tail(x, probs=(0.05, 0.95), c=5.0):
    """Tail-ESS (slide 21): the minimum quantile-ESS over `probs`.

    Use it for quantiles and interval endpoints. Vehtari et al. (2021) take
    the 5% and 95% quantiles; pass e.g. probs=(0.16, 0.84) for the endpoints
    of a 68% interval. Same input shapes as `ess_bulk`.
    """
    return float(min(ess_quantile(x, p, c) for p in probs))


# ===========================================================================
#  Gelman-Rubin split-Rhat  (slide 20)
# ===========================================================================
def _rhat_core(split):
    """Rhat for an (m, n) block of equal-length chains."""
    m, n = split.shape
    chain_means = split.mean(axis=1)
    chain_vars = split.var(axis=1, ddof=1)
    B = n * chain_means.var(ddof=1)         # between-chain variance
    W = chain_vars.mean()                    # within-chain variance
    if W == 0.0:
        return np.inf
    var_plus = (n - 1) / n * W + B / n       # marginal posterior variance
    return float(np.sqrt(var_plus / W))


def _split_rhat_1d(chains):
    """Split-Rhat for one parameter; chains has shape (m, n)."""
    chains = np.asarray(chains, dtype=float)
    m, n = chains.shape
    if m < 2 or n < 4:
        raise ValueError("need >= 2 chains of length >= 4 for split-Rhat")
    half = n // 2
    # split each chain in two -> 2m half-chains; detects early-vs-late drift
    split = np.concatenate([chains[:, :half], chains[:, half:2 * half]], axis=0)
    return _rhat_core(split)


def split_rhat(chains):
    """Split-Rhat. Accepts (n_chains, n_steps) -> scalar, or
    (n_chains, n_steps, n_params) -> array over parameters.
    """
    chains = np.asarray(chains, dtype=float)
    if chains.ndim == 2:
        return _split_rhat_1d(chains)
    if chains.ndim == 3:
        return np.array([_split_rhat_1d(chains[:, :, p])
                         for p in range(chains.shape[2])])
    raise ValueError("split_rhat expects a 2-D or 3-D array")


def _rank_normalize(chains):
    """Blom rank-normalization of pooled samples, reshaped to (m, n)."""
    m, n = chains.shape
    flat = chains.ravel()
    ranks = rankdata(flat)                    # average ranks 1..S
    S = flat.size
    z = norm.ppf((ranks - 3.0 / 8.0) / (S - 0.25))
    return z.reshape(m, n)


def _fold(chains):
    """Fold about the pooled median: |x - median|. Turns a scale (or tail)
    difference between chains into a location difference."""
    return np.abs(chains - np.median(chains))


def rank_normalized_rhat(chains):
    """Rank-normalized split-Rhat (bulk-Rhat, Vehtari et al. 2021): robust to
    heavy tails and non-normal marginals. Same input shapes as split_rhat.
    Sensitive to differences in LOCATION between chains.
    """
    chains = np.asarray(chains, dtype=float)
    if chains.ndim == 2:
        return _split_rhat_1d(_rank_normalize(chains))
    if chains.ndim == 3:
        return np.array([_split_rhat_1d(_rank_normalize(chains[:, :, p]))
                         for p in range(chains.shape[2])])
    raise ValueError("rank_normalized_rhat expects a 2-D or 3-D array")


def folded_rank_normalized_rhat(chains):
    """Folded rank-normalized split-Rhat (Vehtari et al. 2021, slide 20).

    The chains are folded about the pooled median, |x - median|, then
    rank-normalized and split. Chains with the same centre but different
    spread (or tails) look identical to split_rhat / rank_normalized_rhat;
    after folding, that scale mismatch becomes a location mismatch. The
    recommended summary is max(rank_normalized_rhat, folded_rank_normalized_rhat).
    Same input shapes as split_rhat.
    """
    chains = np.asarray(chains, dtype=float)
    if chains.ndim == 2:
        return _split_rhat_1d(_rank_normalize(_fold(chains)))
    if chains.ndim == 3:
        return np.array([_split_rhat_1d(_rank_normalize(_fold(chains[:, :, p])))
                         for p in range(chains.shape[2])])
    raise ValueError("folded_rank_normalized_rhat expects a 2-D or 3-D array")


# ===========================================================================
#  Power spectrum and the Dunkley test  (slides 22-24)
# ===========================================================================
def power_spectrum(x):
    """One-sided periodogram of a centered 1-D chain (slide 22).

    a_j = (1/sqrt(N)) sum_n (x_n - xbar) exp(2 pi i j n / N),  Phat_j = |a_j|^2.
    Returns (j, Phat) for modes j = 1 .. N//2 (j=0 removed by centering).
    """
    x = np.asarray(x, dtype=float)
    N = x.size
    a = np.fft.rfft(x - x.mean()) / np.sqrt(N)
    P = np.abs(a) ** 2
    j = np.arange(P.size)
    return j[1:], P[1:]


def dunkley_model(j, P0, jstar, alpha):
    """Dunkley spectrum P(j) = P0 / (1 + (j/j*)^alpha) (slide 24).

    Works in mode index j, equivalent to k since k_j = 2 pi j / N.
    """
    return P0 / (1.0 + (np.asarray(j, dtype=float) / jstar) ** alpha)


@dataclass
class DunkleyResult:
    """Fit of P(k) = P0 / (1 + (k/k*)^alpha) plus convergence flags (slide 24)."""
    P0: float
    jstar: float
    alpha: float
    r: float                 # = P0 / (N s^2) = 1 / N_eff
    var_mean: float          # Var(xbar) ~ P0 / N
    jstar_ok: bool           # j* > 20
    r_ok: bool               # r < 0.01
    converged: bool          # both
    j: np.ndarray = field(repr=False, default=None)            # periodogram
    power: np.ndarray = field(repr=False, default=None)
    j_binned: np.ndarray = field(repr=False, default=None)     # fit points
    power_binned: np.ndarray = field(repr=False, default=None)

    def fit(self, j):
        """The fitted spectrum P0 / (1 + (j/j*)^alpha) evaluated at modes j."""
        return dunkley_model(j, self.P0, self.jstar, self.alpha)


def dunkley_test(x, n_bins=40, jstar_min=20.0, r_max=0.01):
    """Dunkley et al. power-spectrum diagnostic for one parameter (slides 22-24).

    The raw periodogram is exponentially distributed and very noisy, so it is
    averaged into log-spaced bins (unbiased, low-variance) before the model
    P(k) = P0 / (1 + (k/k*)^alpha) is fitted. Working in mode index j is
    equivalent to k since k_j = 2 pi j / N, so j* is reported directly.

    Convergence requires j* > jstar_min AND r < r_max, where
    r = P0 / (N s^2) = 1 / N_eff. Thresholds are paper-specific (slide 24).
    """
    x = np.asarray(x, dtype=float)
    N = x.size
    s2 = np.var(x, ddof=1)
    j, P = power_spectrum(x)
    jmax = int(j[-1])

    # --- average the periodogram into log-spaced bins -----------------------
    edges = np.unique(
        np.round(np.logspace(0, np.log10(jmax), n_bins + 1)).astype(int)
    )
    jb, Pb, cb = [], [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (j >= lo) & (j <= hi) if i == len(edges) - 2 else (j >= lo) & (j < hi)
        c = int(mask.sum())
        if c:
            jb.append(j[mask].mean())
            Pb.append(P[mask].mean())
            cb.append(c)
    jb, Pb, cb = np.array(jb), np.array(Pb), np.array(cb)

    # --- fit the Dunkley model ---------------------------------------------
    model = dunkley_model

    P0_guess = float(np.mean(Pb[:3])) if Pb.size >= 3 else float(Pb[0])
    sigma = Pb / np.sqrt(cb)                  # SE of a bin mean (exp: sd ~ mean)
    try:
        popt, _ = curve_fit(
            model, jb, Pb, p0=[P0_guess, 20.0, 2.0], sigma=sigma,
            bounds=([0.0, 1e-3, 0.0], [np.inf, np.inf, 12.0]), maxfev=20000,
        )
        P0, jstar, alpha = (float(v) for v in popt)
    except Exception:                        # flat spectrum / no turnover
        P0, jstar, alpha = P0_guess, np.inf, 0.0

    var_mean = P0 / N
    r = P0 / (N * s2) if s2 > 0 else np.inf
    jstar_ok = jstar > jstar_min
    r_ok = r < r_max
    return DunkleyResult(
        P0=P0, jstar=jstar, alpha=alpha, r=r, var_mean=var_mean,
        jstar_ok=jstar_ok, r_ok=r_ok, converged=bool(jstar_ok and r_ok),
        j=j, power=P, j_binned=jb, power_binned=Pb,
    )


# ===========================================================================
#  Report summary  (milestone 4)
# ===========================================================================
def summary(chains, param_names=None):
    """Per-parameter diagnostics table for the convergence report.

    chains: (n_chains, n_steps, n_params). Returns a list of dicts with
    mean, std, split-Rhat, rank-normalized and folded Rhat, ESS (plain, bulk
    and tail), MCSE and the Dunkley flags (evaluated on the first chain, as
    the test is applied per chain).
    """
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 3:
        raise ValueError("summary expects (n_chains, n_steps, n_params)")
    n_params = chains.shape[2]
    names = param_names or [f"theta[{p}]" for p in range(n_params)]

    rows = []
    for p in range(n_params):
        block = chains[:, :, p]              # (n_chains, n_steps)
        dk = dunkley_test(block[0])
        rows.append({
            "name": names[p],
            "mean": float(block.mean()),
            "std": float(block.std(ddof=1)),
            "split_rhat": split_rhat(block),
            "rank_rhat": rank_normalized_rhat(block),
            "folded_rhat": folded_rank_normalized_rhat(block),
            "ess": ess(block),
            "ess_bulk": ess_bulk(block),
            "ess_tail": ess_tail(block),
            "mcse": mcse(block),
            "dunkley_jstar": dk.jstar,
            "dunkley_r": dk.r,
            "dunkley_converged": dk.converged,
        })
    return rows