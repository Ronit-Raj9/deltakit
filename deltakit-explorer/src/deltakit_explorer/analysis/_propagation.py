# (c) Copyright Riverlane 2020-2025.
"""Gaussian linear error propagation for analysis quantities.

Uses the uncertainties package internally. Input standard deviations
(e.g. from calculate_lep_and_lep_stddev) are treated as independent
unless a fit covariance matrix is supplied.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from uncertainties import correlated_values, ufloat
from uncertainties.umath import exp as uexp
from uncertainties.umath import log as ulog


def _nominal_and_std(quantity: ufloat) -> tuple[float, float]:
    return float(quantity.nominal_value), float(quantity.std_dev)


def _uncertain_from_covariance(
    nominal: Sequence[float],
    cov: npt.NDArray[np.floating],
) -> list[ufloat]:
    if len(nominal) != cov.shape[0]:
        msg = (
            f"Length of nominal values ({len(nominal)}) does not match "
            f"covariance matrix size ({cov.shape[0]})."
        )
        raise ValueError(msg)
    return list(correlated_values(nominal, cov))


def leppr_from_single_point(lep: float, lep_stddev: float, rounds: int) -> tuple[float, float]:
    """Propagate uncertainty for single-point LEPPR (Eq. 4, arXiv:2310.05900).

    Args:
        lep: Logical error probability.
        lep_stddev: Standard deviation of the logical error probability.
        rounds: Number of QEC rounds.

    Returns:
        Tuple of LEPPR and its standard deviation.
    """
    uncertain_lep = ufloat(lep, lep_stddev)
    leppr = (1 - (1 - 2 * uncertain_lep) ** (1 / rounds)) / 2
    return _nominal_and_std(leppr)


def log_fidelity_stddev(
    lep: npt.NDArray[np.floating] | Sequence[float],
    lep_stddev: npt.NDArray[np.floating] | Sequence[float],
) -> npt.NDArray[np.float64]:
    """Standard deviation of log(1 - 2*lep) for weighted least-squares.

    Args:
        lep: Logical error probabilities.
        lep_stddev: Standard deviation of each logical error probability.

    Returns:
        Standard deviation of log-fidelity for each input point.
    """
    lep_arr = np.asarray(lep, dtype=np.float64)
    std_arr = np.asarray(lep_stddev, dtype=np.float64)
    return np.asarray(
        [
            float(ulog(1 - 2 * ufloat(float(p), float(s))).std_dev)
            for p, s in zip(lep_arr.ravel(), std_arr.ravel(), strict=True)
        ],
        dtype=np.float64,
    ).reshape(lep_arr.shape)


def epsilon_and_spam_from_log_fit(
    slope: float,
    offset: float,
    cov: npt.NDArray[np.floating],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """LEPPR and SPAM error from correlated log-linear fit parameters.

    Args:
        slope: Slope from log-fidelity linear fit.
        offset: Offset from log-fidelity linear fit.
        cov: Covariance matrix of fit parameters.

    Returns:
        ``((leppr, leppr_stddev), (spam_error, spam_error_stddev))``.
    """
    uncertain_slope, uncertain_offset = _uncertain_from_covariance(
        [slope, offset], cov
    )
    uncertain_leppr = (1 - uexp(uncertain_slope)) / 2
    uncertain_spam = (1 - uexp(uncertain_offset)) / 2
    return _nominal_and_std(uncertain_leppr), _nominal_and_std(uncertain_spam)


def lambda_from_shifted_fit(
    slope: float,
    offset: float,
    cov: npt.NDArray[np.floating],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Error suppression factors from shifted-distance linear fit.

    Args:
        slope: Slope from shifted linear fit.
        offset: Offset from shifted linear fit.
        cov: Covariance matrix of fit parameters.

    Returns:
        ``((lambda_, lambda_std), (lambda0, lambda0_std))``.
    """
    uncertain_slope, uncertain_offset = _uncertain_from_covariance(
        [slope, offset], cov
    )
    uncertain_lambda = uexp(-2 * uncertain_slope)
    uncertain_lambda0 = uexp(-uncertain_offset - ulog(uncertain_lambda) / 2)
    return _nominal_and_std(uncertain_lambda), _nominal_and_std(uncertain_lambda0)


def lambda_from_lin_fit(
    slope: float,
    offset: float,
    cov: npt.NDArray[np.floating],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Error suppression factors from (d+1)/2 linear fit.

    Args:
        slope: Slope from linear fit over ``(d+1)/2``.
        offset: Offset from linear fit over ``(d+1)/2``.
        cov: Covariance matrix of fit parameters.

    Returns:
        ``((lambda_, lambda_std), (lambda0, lambda0_std))``.
    """
    uncertain_slope, uncertain_offset = _uncertain_from_covariance(
        [slope, offset], cov
    )
    uncertain_lambda = uexp(-uncertain_slope)
    uncertain_lambda0 = uexp(-uncertain_offset)
    return _nominal_and_std(uncertain_lambda), _nominal_and_std(uncertain_lambda0)


def lambda_from_curve_fit(
    lamb0: float,
    lamb: float,
    cov: npt.NDArray[np.floating],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Error suppression factors from non-linear curve_fit.

    Args:
        lamb0: Fitted lambda prefactor.
        lamb: Fitted error suppression factor.
        cov: Covariance matrix of fit parameters.

    Returns:
        ``((lambda_, lambda_std), (lambda0, lambda0_std))``.
    """
    uncertain_lamb0, uncertain_lamb = _uncertain_from_covariance([lamb0, lamb], cov)
    return _nominal_and_std(uncertain_lamb), _nominal_and_std(uncertain_lamb0)


def polynomial_derivative_stddev(
    coefficients: npt.NDArray[np.floating],
    cov: npt.NDArray[np.floating],
    point: float,
) -> tuple[float, float]:
    """Gradient and its standard deviation from a fitted polynomial.

    Args:
        coefficients: Polynomial coefficients with index matching power.
        cov: Covariance matrix of polynomial coefficients.
        point: Point at which to evaluate the derivative.

    Returns:
        Tuple of derivative value and its standard deviation.
    """
    uncertain_coefficients = _uncertain_from_covariance(coefficients, cov)
    derivative = sum(
        (power + 1) * coeff * point**power
        for power, coeff in enumerate(uncertain_coefficients[1:])
    )
    return _nominal_and_std(derivative)


def reciprocal_stddev(value: float, stddev: float) -> float:
    """Standard deviation of 1/value.

    Args:
        value: Nominal value.
        stddev: Standard deviation of the nominal value.

    Returns:
        Standard deviation of the reciprocal.
    """
    uncertain_value = ufloat(value, stddev)
    return float((1 / uncertain_value).std_dev)
