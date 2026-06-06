from collections.abc import Sequence
from typing import overload

import numpy as np
import numpy.typing as npt
from deltakit_core.plotting.colours import RIVERLANE_PLOT_COLOURS
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from deltakit_explorer.analysis import LogicalErrorProbabilityPerRoundData as LEPPRData
from deltakit_explorer.plotting._helpers import draw_fit, setup_fig_ax
from deltakit_explorer.plotting.results import (
    LogicalErrorProbabilityPerRoundResult as LEPPRResult,
)
from deltakit_explorer.plotting.results import interpolate_leppr


@overload
def plot_logical_error_probability_per_round(
    leppr_data: LEPPRResult,
    *,
    num_sigmas: int = 3,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]: ...


@overload
def plot_logical_error_probability_per_round(
    leppr_data: LEPPRData,
    num_rounds: npt.NDArray[np.int_] | Sequence[int],
    logical_error_probability: npt.NDArray[np.float64] | Sequence[float],
    logical_error_probability_stddev: (
        npt.NDArray[np.float64] | Sequence[float] | None
    ) = None,
    *,
    num_sigmas: int = 3,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]: ...


def plot_logical_error_probability_per_round(
    leppr_data: LEPPRData | LEPPRResult,
    num_rounds: npt.NDArray[np.int_] | Sequence[int] | None = None,
    logical_error_probability: npt.NDArray[np.float64] | Sequence[float] | None = None,
    logical_error_probability_stddev: (
        npt.NDArray[np.float64] | Sequence[float] | None
    ) = None,
    *,
    num_sigmas: int = 3,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot a logical error probability per round fit and its confidence band.

    When ``leppr_data`` is raw :class:`LogicalErrorProbabilityPerRoundData`, the
    measured data points are drawn and the fit is interpolated; ``num_rounds`` and
    ``logical_error_probability`` are required in that case. When it is an already
    interpolated result, it is rendered directly.

    Args:
        leppr_data: A LEPPR fit, either as raw data or an interpolated result.
        num_rounds: The number of rounds for each measured point. Required for raw
            data, ignored for an interpolated result.
        logical_error_probability: The measured logical error probabilities.
            Required for raw data, ignored for an interpolated result.
        logical_error_probability_stddev: The standard deviation of each measured
            point. If None, no error bars are drawn on the data points.
        num_sigmas: Number of sigmas for the error bars and band.
        fig: A matplotlib Figure to plot on. If None, a new figure is created.
        ax: A matplotlib Axes to plot on. If None, a new axes is created.
        title: An optional custom title for the plot.

    Returns:
        The matplotlib Figure and Axes objects containing the plot.

    Raises:
        ValueError: If raw data is passed without ``num_rounds`` and
            ``logical_error_probability``, or if the data lengths do not match.
    """
    fig, ax = setup_fig_ax(fig, ax)

    if isinstance(leppr_data, LEPPRResult):
        result = leppr_data
    else:
        if num_rounds is None or logical_error_probability is None:
            msg = (
                "num_rounds and logical_error_probability are required when plotting "
                "raw logical error probability per round data."
            )
            raise ValueError(msg)

        lens = {len(num_rounds), len(logical_error_probability)}
        if logical_error_probability_stddev is not None:
            lens.add(len(logical_error_probability_stddev))
        if len(lens) > 1:
            msg = (
                "The lengths of 'num_rounds', 'logical_error_probability' and "
                "'logical_error_probability_stddev' must be the same. Got the "
                f"following lengths: {lens}."
            )
            raise ValueError(msg)

        isort = np.argsort(num_rounds)
        num_rounds = np.asarray(num_rounds)[isort]
        logical_error_probability = np.asarray(logical_error_probability)[isort]
        yerr = (
            None
            if logical_error_probability_stddev is None
            else num_sigmas * np.asarray(logical_error_probability_stddev)[isort]
        )
        ax.errorbar(
            num_rounds,
            logical_error_probability,
            yerr=yerr,
            fmt=".",
            color=RIVERLANE_PLOT_COLOURS[0],
            label=f"Logical error probabilities (±{num_sigmas}σ)",  # noqa: RUF001
        )
        result = interpolate_leppr(leppr_data, num_sigmas=num_sigmas)

    draw_fit(ax, result.rounds, result)
    ax.set_title(title if title is not None else "Logical Error Probability per Round")
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Logical Error Probability")
    return fig, ax
