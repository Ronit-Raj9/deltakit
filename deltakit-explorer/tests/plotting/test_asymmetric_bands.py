import matplotlib as mpl

mpl.use("Agg")

import numpy as np
import pytest

from deltakit_explorer.analysis import (
    calculate_lambda_and_lambda_stddev,
    calculate_lep_and_lep_stddev,
    compute_logical_error_per_round,
)
from deltakit_explorer.plotting import (
    interpolate_lambda,
    interpolate_leppr,
    plot_lambda,
    plot_logical_error_probability_per_round,
)


@pytest.fixture
def leppr_data():
    # Generate counts from a known model so that the largest-round error rate is
    # close to 0.4 and the fit is clean. This keeps compute_logical_error_per_round
    # from warning (which would be turned into an error by the test configuration).
    eps_true, spam_true, shots_per_point = 0.05, 0.01, 200_000
    rounds = np.array([2, 6, 10, 14])
    fidelity = (1 - 2 * spam_true) * (1 - 2 * eps_true) ** rounds
    fails = np.round((1 - fidelity) / 2 * shots_per_point).astype(int)
    shots = np.full(len(rounds), shots_per_point)
    lep, lep_stddev = calculate_lep_and_lep_stddev(fails, shots)
    return (
        compute_logical_error_per_round(rounds, lep, lep_stddev),
        fails,
        shots,
        rounds,
    )


@pytest.fixture
def lambda_data():
    return calculate_lambda_and_lambda_stddev(
        distances=[5, 7, 9],
        leppr=[0.15, 0.1, 0.05],
        leppr_std=[0.01, 0.008, 0.005],
    )


class TestLepprBand:
    def test_band_is_ordered_and_clipped(self, leppr_data):
        result = interpolate_leppr(leppr_data[0])
        assert np.all(result.lower_boundary <= result.interpolated)
        assert np.all(result.interpolated <= result.upper_boundary)
        assert np.all(result.lower_boundary >= 0)
        assert np.all(result.upper_boundary <= 1)

    def test_band_is_asymmetric(self, leppr_data):
        result = interpolate_leppr(leppr_data[0])
        lower_margin = result.interpolated - result.lower_boundary
        upper_margin = result.upper_boundary - result.interpolated
        # The two margins should not be identical everywhere.
        assert not np.allclose(lower_margin, upper_margin)


class TestLambdaBand:
    def test_band_is_ordered_and_positive(self, lambda_data):
        result = interpolate_lambda(lambda_data)
        assert np.all(result.lower_boundary <= result.interpolated)
        assert np.all(result.interpolated <= result.upper_boundary)
        assert np.all(result.lower_boundary >= 0)

    def test_shapes_match(self, lambda_data):
        result = interpolate_lambda(lambda_data, num_points=50)
        assert result.interpolated.shape == result.distances.shape == (50,)


class TestPlotsRun:
    def test_leppr_plot_runs_with_symmetric_bars(self, leppr_data):
        data, _fails, _shots, rounds = leppr_data
        lep, lep_stddev = calculate_lep_and_lep_stddev(_fails, _shots)
        fig, _ = plot_logical_error_probability_per_round(data, rounds, lep, lep_stddev)
        assert fig is not None

    def test_leppr_plot_runs_with_binomial_bars(self, leppr_data):
        data, fails, shots, rounds = leppr_data
        lep, _ = calculate_lep_and_lep_stddev(fails, shots)
        fig, _ = plot_logical_error_probability_per_round(
            data, rounds, lep, num_failed_shots=fails, num_shots=shots
        )
        assert fig is not None

    def test_lambda_plot_runs(self, lambda_data):
        fig, _ = plot_lambda(lambda_data)
        assert fig is not None
