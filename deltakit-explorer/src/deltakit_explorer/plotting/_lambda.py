from deltakit_core.plotting.colours import RIVERLANE_PLOT_COLOURS
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from deltakit_explorer.analysis import LambdaData
from deltakit_explorer.plotting._helpers import draw_fit, setup_fig_ax
from deltakit_explorer.plotting.results import LambdaResult, interpolate_lambda


def plot_lambda(
    lambda_data: LambdaData | LambdaResult,
    *,
    num_sigmas: int = 3,
    num_points: int = 200,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot a Λ fit and its confidence band.

    When ``lambda_data`` is raw :class:`LambdaData`, the per-distance error
    probabilities are drawn and the fit is interpolated. When it is an already
    interpolated :class:`LambdaResult`, it is rendered directly.

    Args:
        lambda_data: A Λ fit, either as raw data or an interpolated result.
        num_sigmas: Number of standard deviations for the error band. Default 3.
        num_points: Number of interpolation points. Default 200.
        fig: A matplotlib Figure to plot on. If None, a new figure is created.
        ax: A matplotlib Axes to plot on. If None, a new axes is created.
        title: An optional custom title for the plot.

    Returns:
        The matplotlib Figure and Axes objects containing the plot.

    Example:
        from deltakit_explorer.analysis import calculate_lambda_and_lambda_stddev

        lambda_data = calculate_lambda_and_lambda_stddev(
            distances=[5, 7, 9],
            leppr=[0.15, 0.1, 0.05],
            leppr_std=[0.01, 0.008, 0.005],
        )
        fig, ax = plot_lambda(lambda_data)
        ax.set_yscale("log")
    """
    fig, ax = setup_fig_ax(fig, ax)

    if isinstance(lambda_data, LambdaResult):
        result = lambda_data
    else:
        ax.errorbar(
            lambda_data.distances,
            lambda_data.leppr,
            yerr=lambda_data.leppr_std * num_sigmas,
            fmt=".",
            color=RIVERLANE_PLOT_COLOURS[1],
            label=f"Logical error probabilities per round (±{num_sigmas}σ)",  # noqa: RUF001
        )
        result = interpolate_lambda(
            lambda_data, num_sigmas=num_sigmas, num_points=num_points
        )

    draw_fit(ax, result.distances, result)
    ax.set_title(title if title is not None else "Error Suppression Factor Λ")
    ax.set_xlabel("Code distance")
    ax.set_ylabel("Logical Error Probability per Round")
    return fig, ax
