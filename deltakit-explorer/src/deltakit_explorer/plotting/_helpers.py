# (c) Copyright Riverlane 2020-2025.
"""Shared rendering helpers for the specialised plotting functions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from deltakit_core.plotting.colours import RIVERLANE_PLOT_COLOURS
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from deltakit_explorer.plotting.results import Interpolated


def setup_fig_ax(fig: Figure | None, ax: Axes | None) -> tuple[Figure, Axes]:
    """Validate a ``fig`` / ``ax`` pair, creating them if both are None.

    Args:
        fig: An existing matplotlib Figure, or None.
        ax: An existing matplotlib Axes, or None.

    Returns:
        The figure and axes to draw on.

    Raises:
        ValueError: If exactly one of ``fig`` and ``ax`` is None.
    """
    if (fig is None) ^ (ax is None):
        msg = "The 'fig' and 'ax' parameters should either be both `None` or both set."
        raise ValueError(msg)
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    return fig, ax


def draw_fit(
    ax: Axes, x_values: npt.NDArray[np.floating], result: Interpolated
) -> None:
    """Draw an interpolated fit curve and its confidence band onto ``ax``.

    Args:
        ax: The axes to draw on.
        x_values: The x coordinates of the interpolated curve.
        result: The interpolated values and confidence boundaries to draw.
    """
    ax.plot(
        x_values,
        result.interpolated,
        label=result.fit_label,
        color=RIVERLANE_PLOT_COLOURS[1],
    )
    ax.fill_between(
        x_values,
        result.lower_boundary,
        result.upper_boundary,
        label=result.confidence_interval_label,
        color=RIVERLANE_PLOT_COLOURS[0],
        alpha=0.2,
    )
    ax.legend()
