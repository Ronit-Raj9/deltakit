import math

import numpy as np
import pytest

from deltakit_explorer.analysis import (
    Fit,
    calculate_lep_asymmetric,
    fit_binomial,
    fit_binomial_batch,
    log_binomial,
)


class TestLogBinomial:
    def test_matches_sinter_documented_values(self) -> None:
        # From sinter._probability_util.log_binomial docstring examples.
        assert log_binomial(p=0.5, n=100, hits=50) == pytest.approx(
            -2.5308762, abs=1e-5
        )
        assert log_binomial(p=0.2, n=1_000_000, hits=1_000) == pytest.approx(
            -216626.97, rel=1e-6
        )

    def test_zero_probability_with_hits_is_neg_inf(self) -> None:
        assert log_binomial(p=0.0, n=10, hits=1) == -math.inf

    def test_certain_probability_with_misses_is_neg_inf(self) -> None:
        assert log_binomial(p=1.0, n=10, hits=1) == -math.inf


class TestFitBinomial:
    def test_matches_sinter_balanced_example(self) -> None:
        # sinter.fit_binomial(num_shots=10, num_hits=5, factor=9) -> 0.202/0.5/0.798
        fit = fit_binomial(num_shots=10, num_hits=5, max_likelihood_factor=9.0)
        assert fit.best == pytest.approx(0.5)
        assert fit.low == pytest.approx(0.202, abs=1e-3)
        assert fit.high == pytest.approx(0.798, abs=1e-3)

    def test_upward_skewed_near_zero(self) -> None:
        # sinter.fit_binomial(1e8, 2, 1000) -> low=2e-10, best=2e-8, high=1.259e-7
        fit = fit_binomial(num_shots=100_000_000, num_hits=2)
        assert fit.best == pytest.approx(2e-8)
        assert fit.high == pytest.approx(1.259e-7, rel=1e-3)
        assert fit.upper_margin > 5 * fit.lower_margin

    def test_low_bound_is_non_negative(self) -> None:
        fit = fit_binomial(num_shots=1_000_000, num_hits=1)
        assert fit.low >= 0

    def test_zero_hits(self) -> None:
        fit = fit_binomial(num_shots=1_000_000, num_hits=0)
        assert fit.low == 0.0
        assert fit.best == 0.0
        assert fit.high > 0

    def test_all_hits(self) -> None:
        fit = fit_binomial(num_shots=100, num_hits=100)
        assert fit.high == 1.0
        assert fit.best == 1.0

    def test_zero_shots(self) -> None:
        assert fit_binomial(num_shots=0, num_hits=0) == Fit(low=0.0, best=0.5, high=1.0)

    @pytest.mark.parametrize(
        ("shots", "hits", "factor"),
        [(10, 11, 1000.0), (-1, 0, 1000.0), (10, 5, 0.5)],
    )
    def test_invalid_inputs_raise(self, shots: int, hits: int, factor: float) -> None:
        with pytest.raises(ValueError):
            fit_binomial(num_shots=shots, num_hits=hits, max_likelihood_factor=factor)


class TestFitBinomialBatch:
    def test_shapes_and_values_match_scalar(self) -> None:
        shots = np.array([1000, 5000])
        hits = np.array([3, 40])
        low, best, high = fit_binomial_batch(shots, hits)
        assert low.shape == best.shape == high.shape == (2,)
        scalar = fit_binomial(num_shots=1000, num_hits=3)
        assert low[0] == pytest.approx(scalar.low)
        assert high[0] == pytest.approx(scalar.high)

    def test_mismatched_shapes_raise(self) -> None:
        with pytest.raises(ValueError):
            fit_binomial_batch([1000, 2000], [3])


class TestCalculateLepAsymmetric:
    def test_best_is_ratio(self) -> None:
        low, best, high = calculate_lep_asymmetric([2, 151, 34], [500000] * 3)
        np.testing.assert_allclose(best, np.array([2, 151, 34]) / 500000)
        assert np.all(low >= 0)
        assert np.all(low <= best)
        assert np.all(best <= high)

    def test_asymmetric_when_few_hits(self) -> None:
        low, best, high = calculate_lep_asymmetric([2], [1_000_000])
        lower_margin = best[0] - low[0]
        upper_margin = high[0] - best[0]
        assert upper_margin != pytest.approx(lower_margin, rel=0.2)

    def test_accepts_zero_fails(self) -> None:
        low, best, high = calculate_lep_asymmetric([0], [1_000_000])
        assert low[0] == 0.0
        assert best[0] == 0.0
        assert high[0] > 0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            calculate_lep_asymmetric([1, 2], [1000])

    def test_negative_fails_raise(self) -> None:
        with pytest.raises(ValueError):
            calculate_lep_asymmetric([-1], [1000])
