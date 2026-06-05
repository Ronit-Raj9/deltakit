from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from deltakit_core.plotting.colours import RIVERLANE_PLOT_COLOURS
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from deltakit_explorer.analysis import LogicalErrorProbabilityPerRoundData as LEPPRData
from deltakit_explorer.analysis import calculate_lep_asymmetric
from deltakit_explorer.plotting.plotting import plot
from deltakit_explorer.plotting.results import interpolate_leppr


def plot_logical_error_probability_per_round(
    leppr_data: LEPPRData,
    num_rounds: npt.NDArray[np.int_] | Sequence[int],
    logical_error_probability: npt.NDArray[np.float64] | Sequence[float],
    logical_error_probability_stddev: (
        npt.NDArray[np.float64] | Sequence[float] | None
    ) = None,
    *,
    num_sigmas: int = 3,
    num_failed_shots: npt.NDArray[np.int_] | Sequence[int] | None = None,
    num_shots: npt.NDArray[np.int_] | Sequence[int] | None = None,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot the logical error probability per round data and the fitted curve.

    Args:
        leppr_data: Data class containing logical error probability per round
            fit results.
        num_rounds: a sequence of integers representing the number of rounds
            used to get the corresponding results in ``num_failed_shots`` and
            ``num_shots``.
        logical_error_probability: a sequence of floats representing the logical
            error probabilities corresponding to the number of rounds in
            ``num_rounds``.
        logical_error_probability_stddev: a sequence of floats representing the
            standard deviation of the logical error probabilities corresponding
            to the number of rounds in ``num_rounds``. If None, no error bars
            will be plotted. Default is None.
        num_sigmas: number of sigmas to consider when plotting the symmetric
            error bars and the fitted confidence band.
        num_failed_shots: the number of logical failures per round count. When
            given together with ``num_shots``, the data points are drawn with
            asymmetric binomial error bars instead of the symmetric standard
            deviation. Default is None.
        num_shots: the number of shots per round count, used together with
            ``num_failed_shots`` for the binomial error bars. Default is None.
        fig: a matplotlib Figure object to plot on. If None, a new figure
            will be created. Default is None.
        ax: a matplotlib Axes object to plot on. If None, a new axes will
            be created. Default is None.

    Returns:
        The matplotlib Figure and Axes objects containing the plot.

    Example:

        >>> from deltakit_explorer.analysis import (
        ...     calculate_lep_and_lep_stddev,
        ...     compute_logical_error_per_round,
        ... )
        >>> num_failed_shots = [34, 151, 356]
        >>> num_shots = [500000] * 3
        >>> num_rounds = [2, 4, 6]
        >>> res = compute_logical_error_per_round(
        ...     num_failed_shots=num_failed_shots,
        ...     num_shots=num_shots,
        ...     num_rounds=num_rounds,
        ... )
        >>> lep, lep_stddev = calculate_lep_and_lep_stddev(
        ...     fails=num_failed_shots, shots=num_shots
        ... )
        >>> fig, ax = plot_logical_error_probability_per_round(
        ...     res,
        ...     num_rounds=num_rounds,
        ...     logical_error_probability=lep,
        ...     logical_error_probability_stddev=lep_stddev,
        ... )
    """
    if (fig is None) ^ (ax is None):
        msg = "The 'fig' and 'ax' parameters should either be both None or both set."
        raise ValueError(msg)

    if fig is None and ax is None:
        fig, ax = plt.subplots()

    assert ax is not None
    assert fig is not None

    lens = {len(num_rounds), len(logical_error_probability)}
    if logical_error_probability_stddev is not None:
        lens.add(len(logical_error_probability_stddev))
    if len(lens) > 1:
        msg = (
            "The lengths of 'num_rounds', 'logical_error_probability' and "
            "'logical_error_probability_stddev' must be the same. Got the following "
            f"lengths: {lens}."
        )
        raise ValueError(msg)

    isort = np.argsort(num_rounds)
    num_rounds = np.asarray(num_rounds)[isort]
    logical_error_probability = np.asarray(logical_error_probability)[isort]

    # Use a binomial interval for the data points when the raw counts are given,
    # otherwise fall back to the symmetric standard deviation.
    if num_failed_shots is not None and num_shots is not None:
        low, _, high = calculate_lep_asymmetric(
            np.asarray(num_failed_shots)[isort], np.asarray(num_shots)[isort]
        )
        yerr = np.clip(
            np.vstack(
                (logical_error_probability - low, high - logical_error_probability)
            ),
            0,
            None,
        )
        marker_label = "Logical error probabilities (binomial interval)"
    elif logical_error_probability_stddev is not None:
        yerr = num_sigmas * np.asarray(logical_error_probability_stddev)[isort]
        marker_label = f"Logical error probabilities (±{num_sigmas}σ)"  # noqa: RUF001
    else:
        yerr = None
        marker_label = "Logical error probabilities"

    ax.errorbar(
        num_rounds,
        logical_error_probability,
        yerr=yerr,
        fmt=".",
        color=RIVERLANE_PLOT_COLOURS[0],
        label=marker_label,
    )

    leppr_result = interpolate_leppr(leppr_data, num_sigmas=num_sigmas)

    plot(leppr_result, fig=fig, ax=ax)

    return fig, ax
