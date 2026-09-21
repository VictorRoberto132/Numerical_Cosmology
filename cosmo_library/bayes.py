"""
In this file, we define the Bayesian inference methods used in the Cosmo Library.
Posterior, Gaussian likelihood, and prior distributions are implemented here.

Bayesian core: target density, proposals, and Metropolis-Hastings.

Everything is done in log-probability. The sampler is
target-agnostic: it talks to a Posterior object and a Proposal object, so
the very same code samples a toy Gaussian (milestone 2) and the Union2.1
posterior (milestone 3).
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular


# ===========================================================================
#  Target density
# ===========================================================================
class Posterior:
    """Unnormalized log-posterior: log pi(theta) = log L(theta) + log prior(theta).

    Parameters
    ----------
    log_likelihood : callable
        theta -> log L(theta) (or model_instance -> log L if `model` given).
    log_prior : callable
        theta -> log prior(theta); returns -inf outside the support.
    model : callable, optional
        theta -> physical model object. If provided, the likelihood is called
        on model(theta) rather than on theta directly. Kept optional and last
        so a pure-math target (the Gaussian acceptance test) needs no model.

    Notes
    -----
    The prior is evaluated first. If it is -inf the likelihood is never
    called: this both saves work and avoids evaluating an invalid model
    (e.g. E^2(z) < 0) outside the allowed region. Any non-finite likelihood
    is coerced to -inf, so log_prob never returns NaN.
    """

    def __init__(self, log_likelihood, log_prior, model=None):
        self.log_likelihood = log_likelihood
        self.log_prior = log_prior
        self.model = model

    def log_prob(self, theta):
        theta = np.asarray(theta, dtype=float)

        lp = float(self.log_prior(theta))
        if not np.isfinite(lp):
            return -np.inf                      # outside the prior support

        if self.model is not None:
            ll = float(self.log_likelihood(self.model(theta)))
        else:
            ll = float(self.log_likelihood(theta))

        if not np.isfinite(ll):
            return -np.inf                      # invalid model / underflow
        return lp + ll

    # convenience so a Posterior is itself callable
    def __call__(self, theta):
        return self.log_prob(theta)


# ===========================================================================
#  Proposals
# ===========================================================================
class Proposal:
    """Interface for MH proposals: propose + logpdf.

    A proposal must be able to (a) draw a candidate and (b) evaluate the log
    density of a move in BOTH directions, so the sampler handles symmetric
    and asymmetric proposals with one code path.
    """

    def propose(self, theta, rng):
        """Return a candidate theta' ~ q(. | theta)."""
        raise NotImplementedError

    def logpdf(self, theta_prime, theta):
        """Return log q(theta' | theta)."""
        raise NotImplementedError


class GaussianRandomWalk(Proposal):
    """
    Symmetric Gaussian random walk: theta' = theta + L z, z ~ N(0, I),
    with L L^T = cov.

    Accepts a full covariance matrix, a 1-D vector of per-parameter variances
    (treated as a diagonal covariance), or a scalar variance for a 1-D target.
    Because the proposal is symmetric, logpdf cancels in the MH ratio; it is
    still implemented in full so the same sampler accepts asymmetric proposals.
    """

    def __init__(self, cov):
        cov = np.asarray(cov, dtype=float)
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)
        elif cov.ndim == 1:
            cov = np.diag(cov)
        if cov.shape[0] != cov.shape[1]:
            raise ValueError("covariance must be square")

        self.cov = cov
        self.dim = cov.shape[0]
        self.L = np.linalg.cholesky(cov)                 # lower-triangular
        self._log_norm = (0.5 * self.dim * np.log(2.0 * np.pi)
                          + np.sum(np.log(np.diag(self.L))))

    def propose(self, theta, rng):
        theta = np.asarray(theta, dtype=float)
        z = rng.standard_normal(self.dim)
        return theta + self.L @ z

    def logpdf(self, theta_prime, theta):
        d = np.asarray(theta_prime, dtype=float) - np.asarray(theta, dtype=float)
        y = solve_triangular(self.L, d, lower=True)      # y = L^{-1} d
        return -0.5 * (y @ y) - self._log_norm


def scaled_proposal_cov(samples, scale=None, reg=1e-6):
    """Random-walk proposal covariance from a warm-up chain (slides 18-19).

    Returns (2.4^2 / d) * (sample_cov + ridge), the Roberts-Gelman-Gilks rule
    of thumb for a d-dimensional target. `scale` overrides the 2.4^2 / d
    factor.

    The sample covariance is regularized before it is handed to a Cholesky
    factorization (slide 18): it is symmetrized and a ridge of
    `reg * mean(diag)` is added to the diagonal, so a nearly degenerate
    warm-up chain still gives a positive-definite matrix. The ridge is
    relative to the parameter scale, so it does not depend on units. A chain
    that never moved has no covariance to estimate and raises ValueError
    (the proposal was too large or too small during warm-up).

    Estimate this from a warm-up chain, then FREEZE it before the retained
    run: adapting the proposal while collecting samples breaks the Markov
    property that the diagnostics assume.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError("samples must have shape (n_steps, dim)")
    dim = samples.shape[1]
    if scale is None:
        scale = 2.4 ** 2 / dim

    cov = np.atleast_2d(np.cov(samples, rowvar=False))
    cov = 0.5 * (cov + cov.T)
    mean_var = np.trace(cov) / dim
    if not np.isfinite(mean_var) or mean_var <= 0.0:
        raise ValueError(
            "warm-up samples have no spread (the chain never moved); "
            "rescale the warm-up proposal before estimating a covariance"
        )
    return scale * (cov + reg * mean_var * np.eye(dim))


def dispersed_starts(posterior, bounds, n, rng, oversample=50, max_tries=20000):
    """Choose `n` over-dispersed, valid initial states inside the prior box.

    Draws `oversample * n` candidates uniformly in the box, keeping only those
    with finite log-probability (inside the prior AND a valid model, e.g.
    E^2(z) > 0), then picks `n` of them by greedy maximin (farthest-point)
    selection in box-normalized coordinates. Plain uniform draws can cluster
    by chance (e.g. all on one side of the posterior), which would make Rhat
    blind to that direction; maximin guarantees well-separated starts.

    Parameters
    ----------
    posterior : Posterior
    bounds : sequence of (low, high) per parameter
    n : int
        Number of starting points wanted.
    rng : numpy.random.Generator
        Explicit generator, for reproducibility.
    """
    lows = np.array([b[0] for b in bounds], dtype=float)
    highs = np.array([b[1] for b in bounds], dtype=float)
    n_cand = max(int(oversample) * n, n)

    cand = []
    for _ in range(max_tries):
        theta = rng.uniform(lows, highs)
        if np.isfinite(posterior.log_prob(theta)):
            cand.append(theta)
            if len(cand) == n_cand:
                break
    if len(cand) < n:
        raise RuntimeError(
            f"found only {len(cand)}/{n} valid starts in {max_tries} draws; "
            "the prior box may barely overlap the valid region"
        )
    cand = np.array(cand)

    unit = (cand - lows) / (highs - lows)           # box-normalized coordinates
    first = np.argmax(np.sum((unit - unit.mean(axis=0)) ** 2, axis=1))
    chosen = [first]
    dmin = np.sum((unit - unit[first]) ** 2, axis=1)
    for _ in range(n - 1):
        nxt = int(np.argmax(dmin))
        chosen.append(nxt)
        dmin = np.minimum(dmin, np.sum((unit - unit[nxt]) ** 2, axis=1))
    return cand[chosen]


# ===========================================================================
#  Sampler
# ===========================================================================
@dataclass
class SamplingResult:
    """Output of MetropolisHastings.sample."""
    samples: np.ndarray          # (n_steps, dim)
    log_probs: np.ndarray        # (n_steps,)
    accepted: np.ndarray         # (n_steps,) bool
    acceptance_rate: float


class MetropolisHastings:
    """Metropolis--Hastings sampler.

    Cares about the details that make the retained chain correct:
      * everything in log-probability;
      * a non-finite target counts as a rejection;
      * the current log density is cached, not recomputed after a rejection;
      * every iteration is stored, and on rejection the current state is
        appended again -- never keep only accepted proposals;
      * an explicit random generator makes runs reproducible from a seed.
    """

    def __init__(self, posterior, proposal):
        self.posterior = posterior
        self.proposal = proposal

    def sample(self, initial_state, n_steps, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        state = np.asarray(initial_state, dtype=float).copy()
        dim = state.size

        logp = self.posterior.log_prob(state)
        if not np.isfinite(logp):
            raise ValueError(
                "initial state has non-finite log-probability; "
                "start the chain at a valid, in-prior point"
            )

        samples = np.empty((n_steps, dim))
        log_probs = np.empty(n_steps)
        accepted = np.zeros(n_steps, dtype=bool)

        for t in range(n_steps):
            candidate = self.proposal.propose(state, rng)
            logp_new = self.posterior.log_prob(candidate)

            # log alpha = log[ pi(x')/pi(x) ] + log[ q(x|x')/q(x'|x) ]
            log_alpha = logp_new - logp
            log_alpha += self.proposal.logpdf(state, candidate)     # q(x | x')
            log_alpha -= self.proposal.logpdf(candidate, state)     # q(x' | x)

            # log_alpha is -inf when the target is -inf -> guaranteed rejection
            u = rng.random()
            if np.log(u) < min(0.0, log_alpha):
                state = candidate
                logp = logp_new
                accepted[t] = True

            samples[t] = state          # store EVERY step, repeated on reject
            log_probs[t] = logp

        return SamplingResult(
            samples=samples,
            log_probs=log_probs,
            accepted=accepted,
            acceptance_rate=float(accepted.mean()),
        )