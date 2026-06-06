# (c) Copyright Riverlane 2020-2025.
"""Generic dispatch-based plotting interface for deltakit-explorer."""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from deltakit_explorer.plotting._lambda import plot_lambda
from deltakit_explorer.plotting._leppr import (
    plot_logical_error_probability_per_round,
)
from deltakit_explorer.plotting.results import LambdaResult
from deltakit_explorer.plotting.results import (
    LogicalErrorProbabilityPerRoundResult as LEPPRResult,
)


def plot(
    result: LambdaResult | LEPPRResult,
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot a precomputed result by dispatching to the matching specialised plotter.

    ``plot`` inspects the type of ``result`` and forwards it to the function that
    owns the rendering for that type, so a precomputed result (from
    ``interpolate_lambda`` or ``interpolate_leppr``) can be drawn with a single
    call.

    Args:
        result: The precomputed plot data.
        fig: An existing matplotlib Figure. If None, a new figure is created.
        ax: An existing matplotlib Axes. If None, a new axes is created.
        title: An optional custom title for the plot.

    Returns:
        The matplotlib Figure and Axes objects containing the plot.

    Raises:
        TypeError: If the ``result`` type is not supported.

    Examples:
        Plot a precomputed Λ fit::

            from deltakit_explorer.plotting.results import interpolate_lambda

            lambda_result = interpolate_lambda(lambda_data)
            fig, ax = plot(lambda_result)

        Plot a precomputed LEPPR fit::

            from deltakit_explorer.plotting.results import interpolate_leppr

            leppr_result = interpolate_leppr(leppr_data)
            fig, ax = plot(leppr_result)
    """
    match result:
        case LambdaResult():
            return plot_lambda(result, fig=fig, ax=ax, title=title)
        case LEPPRResult():
            return plot_logical_error_probability_per_round(
                result, fig=fig, ax=ax, title=title
            )
        case _:
            msg = (
                f"Unsupported result type: {type(result).__name__}. "
                "Expected `LambdaResult` or `LEPPRResult`."
            )
            raise TypeError(msg)
