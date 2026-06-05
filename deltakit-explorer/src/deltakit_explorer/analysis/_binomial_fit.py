# (c) Copyright Riverlane 2020-2025.
"""Asymmetric binomial confidence intervals for error-rate estimates.

A logical error probability estimated from ``num_hits`` failures out of
``num_shots`` shots follows a binomial distribution. The symmetric Gaussian
interval ``p +/- sqrt(p (1 - p) / n)`` is a poor description of that
distribution when ``p`` is close to zero: it can extend below zero and it
underestimates the upper tail. This module computes the asymmetric interval
instead, following the approach used by Stim's ``sinter``.

``log_binomial`` and the ``fit_binomial`` search are adapted from
``sinter._probability_util`` (Apache-2.0, quantumlib/Stim). See
https://quantumcomputing.stackexchange.com/a/37268 for an interpretation of the
``max_likelihood_factor`` interval.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy import optimize

#: Default Bayes factor used to define the confidence interval. Hypotheses whose
#: likelihood is more than this many times less likely than the best fit are
#: excluded. A factor of 1000 corresponds to roughly a 99% interval.
DEFAULT_MAX_LIKELIHOOD_FACTOR = 1000.0


@dataclasses.dataclass(frozen=True)
class Fit:
    """A point estimate together with its lower and upper bounds.

    Attributes:
        low: Smallest rate compatible with the data at the chosen confidence.
        best: Maximum-likelihood estimate (``num_hits / num_shots``).
        high: Largest rate compatible with the data at the chosen confidence.
    """

    low: float
    best: float
    high: float

    @property
    def lower_margin(self) -> float:
        """How far ``best`` sits above ``low``."""
        return self.best - self.low

    @property
    def upper_margin(self) -> float:
        """How far ``high`` sits above ``best``."""
        return self.high - self.best


def log_binomial(*, p: float, n: int, hits: int) -> float:
    """Return ``ln P(hits | Binomial(n, p))``.

    The computation is done in log space so that the tiny probabilities involved
    in large experiments stay representable.

    Args:
        p: Hypothesis probability, between 0 and 1.
        n: Number of shots.
        hits: Number of failures observed.

    Returns:
        The natural log of the binomial likelihood.
    """
    p = min(max(p, 0.0), 1.0)
    misses = n - hits
    result = 0.0
    if hits:
        if p == 0:
            return -math.inf
        result += math.log(p) * hits
    if misses:
        if p == 1:
            return -math.inf
        result += math.log1p(-p) * misses
    result += math.lgamma(n + 1) - math.lgamma(misses + 1) - math.lgamma(hits + 1)
    return result


def fit_binomial(
    *,
    num_shots: int,
    num_hits: int,
    max_likelihood_factor: float = DEFAULT_MAX_LIKELIHOOD_FACTOR,
) -> Fit:
    """Estimate an error rate and its asymmetric confidence interval.

    The interval contains every rate whose binomial likelihood is within
    ``max_likelihood_factor`` of the most likely rate ``num_hits / num_shots``.

    Args:
        num_shots: Number of shots.
        num_hits: Number of failures observed.
        max_likelihood_factor: How much less likely than the best fit a rate may
            be before it is excluded from the interval. Must be at least 1.

    Returns:
        A :class:`Fit` with the best estimate and its low and high bounds.

    Raises:
        ValueError: If the inputs are out of range.
    """
    if max_likelihood_factor < 1:
        msg = f"max_likelihood_factor={max_likelihood_factor} must be >= 1."
        raise ValueError(msg)
    if num_shots < 0 or num_hits < 0 or num_hits > num_shots:
        msg = f"Need 0 <= num_hits ({num_hits}) <= num_shots ({num_shots})."
        raise ValueError(msg)
    if num_shots == 0:
        return Fit(low=0.0, best=0.5, high=1.0)

    best = num_hits / num_shots
    target = log_binomial(p=best, n=num_shots, hits=num_hits) - math.log(
        max_likelihood_factor
    )

    def gap(p: float) -> float:
        return log_binomial(p=p, n=num_shots, hits=num_hits) - target

    # The likelihood is unimodal, so each tail crosses the target exactly once.
    low = 0.0 if num_hits == 0 else float(optimize.brentq(gap, 1e-18, best))
    high = (
        1.0 if num_hits == num_shots else float(optimize.brentq(gap, best, 1 - 1e-18))
    )
    return Fit(low=low, best=best, high=high)


def fit_binomial_batch(
    num_shots: npt.NDArray[np.int_] | Sequence[int],
    num_hits: npt.NDArray[np.int_] | Sequence[int],
    *,
    max_likelihood_factor: float = DEFAULT_MAX_LIKELIHOOD_FACTOR,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Apply :func:`fit_binomial` to each (shots, hits) pair.

    Args:
        num_shots: Number of shots per point.
        num_hits: Number of failures per point.
        max_likelihood_factor: Passed through to :func:`fit_binomial`.

    Returns:
        Three arrays ``(low, best, high)`` with the same shape as the inputs.

    Raises:
        ValueError: If ``num_shots`` and ``num_hits`` have different shapes.
    """
    shots = np.asarray(num_shots, dtype=np.int_)
    hits = np.asarray(num_hits, dtype=np.int_)
    if shots.shape != hits.shape:
        msg = "num_shots and num_hits must have the same shape."
        raise ValueError(msg)

    fits = [
        fit_binomial(
            num_shots=int(s),
            num_hits=int(h),
            max_likelihood_factor=max_likelihood_factor,
        )
        for s, h in zip(shots.ravel(), hits.ravel(), strict=True)
    ]
    low = np.array([f.low for f in fits], dtype=np.float64).reshape(shots.shape)
    best = np.array([f.best for f in fits], dtype=np.float64).reshape(shots.shape)
    high = np.array([f.high for f in fits], dtype=np.float64).reshape(shots.shape)
    return low, best, high
