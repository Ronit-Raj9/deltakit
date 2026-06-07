from collections.abc import Callable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from deltakit_circuit._circuit import Circuit
from deltakit_decode.analysis import RunAllAnalysisEngine

from deltakit_explorer.analysis import ConfidenceInterval
from deltakit_explorer.analysis.error_budget._generation import (
    generate_decoder_managers_for_lambda,
)
from deltakit_explorer.analysis.error_budget._memory import (
    MemoryGenerator,
    PreComputedMemoryGenerator,
    get_rotated_surface_code_memory_circuit,
)
from deltakit_explorer.analysis.error_budget._parameters import SamplingParameters
from deltakit_explorer.analysis.error_budget._post_processing import (
    _filter_non_close_noise_parameters,
    compute_lambda_and_stddev_from_results,
    compute_lambda_interval_from_results,
)


def _run_lambda_engine(
    noise_model: Callable[[Circuit, npt.NDArray[np.floating]], Circuit],
    noise_parameters: npt.NDArray[np.floating] | Sequence[float],
    num_rounds_by_distances: Mapping[int, Sequence[int]],
    sampling_parameters: SamplingParameters,
    memory_generator: MemoryGenerator | Mapping[int, Mapping[int, Circuit]],
) -> tuple[npt.NDArray[np.floating], list[str], pd.DataFrame]:
    """Sample the memory experiments needed to estimate Λ at a single point.

    Args:
        noise_model: a callable adding noise to a circuit from the parameters.
        noise_parameters: the parameters forwarded to ``noise_model``.
        num_rounds_by_distances: a mapping from each code distance to the number
            of rounds to sample.
        sampling_parameters: parameters relating to the sampling tasks.
        memory_generator: a callable that generates a memory experiment.

    Returns:
        The reshaped noise parameters, their identifier names, and the report
        frame produced by the analysis engine.
    """
    if isinstance(memory_generator, Mapping):
        memory_generator = PreComputedMemoryGenerator(memory_generator)

    point = np.asarray(noise_parameters).reshape((-1, 1))

    # Create unique identifiers for noise parameters that will be used to
    # discriminate between them in the CSV file storing the simulation results.
    noise_parameter_names = [str(i) for i in range(point.size)]

    decoder_managers = generate_decoder_managers_for_lambda(
        point,
        noise_model,
        num_rounds_by_distances,
        sampling_parameters.max_workers,
        memory_generator=memory_generator,
        noise_parameter_names=noise_parameter_names,
    )
    engine = RunAllAnalysisEngine(
        experiment_name="Estimating 1 / Λ",
        decoder_managers=decoder_managers,
        max_shots=sampling_parameters.max_shots,
        batch_size=sampling_parameters.batch_size,
        # Early stopping when we have a low-enough standard deviation
        loop_condition=RunAllAnalysisEngine.loop_until_observable_rse_below_threshold(
            sampling_parameters.lep_target_rse,
            sampling_parameters.lep_computation_min_fails,
        ),
        num_parallel_processes=sampling_parameters.max_workers,
    )
    return point, noise_parameter_names, engine.run()


def inverse_lambda_at(
    noise_model: Callable[[Circuit, npt.NDArray[np.floating]], Circuit],
    noise_parameters: npt.NDArray[np.floating] | Sequence[float],
    num_rounds_by_distances: Mapping[int, Sequence[int]],
    sampling_parameters: SamplingParameters = SamplingParameters(),
    memory_generator: MemoryGenerator
    | Mapping[int, Mapping[int, Circuit]] = get_rotated_surface_code_memory_circuit,
) -> tuple[float, float]:
    """Compute 1 / Λ.

    Warning:
        This is a helper function to compute 1 / Λ when you need a **single**
        evaluation.
        For error budgeting, :func:`~deltakit_explorer.analysis.error_budget.get_error_budget`
        will be able to parallelise more efficiently, while also performing several
        checks and optimisations.

    Args:
        noise_model (Callable[[Circuit, npt.NDArray[np.floating]], Circuit]): a callable
            adding noise to the provided circuit, according to the parameters provided.
        noise_parameters (npt.NDArray[numpy.floating] | Sequence[float]): valid
            parameters to forward to ``noise_model`` representing the point at which the
            gradient should be computed.
        num_rounds_by_distances (Mapping[int, Sequence[int]]): a mapping from each code
            distance that should be tested to the number of rounds that should be
            sampled in order to estimate the logical error-probability per round, to
            ultimately get 1 / Λ.
        sampling_parameters: additional parameters relating to the sampling tasks used to
            estimate 1 / Λ indirectly.
        memory_generator (MemoryGenerator): a callable that can generate a memory
            experiment. The resulting circuit will go through the provided
            ``noise_model`` for different values of the noise parameters.

    Returns:
        the estimation of 1 / Λ along with the standard deviation of the estimation as
        a 2-tuple.
    """
    point, noise_parameter_names, report = _run_lambda_engine(
        noise_model,
        noise_parameters,
        num_rounds_by_distances,
        sampling_parameters,
        memory_generator,
    )
    lambdas, lambda_stddevs = compute_lambda_and_stddev_from_results(
        point, noise_parameter_names, num_rounds_by_distances, report
    )
    lambda_reciprocals = 1 / lambdas
    lambda_reciprocal_stddevs = np.abs(lambda_stddevs / lambdas**2)

    return float(lambda_reciprocals[0, 0]), float(lambda_reciprocal_stddevs[0, 0])


def inverse_lambda_interval_at(
    noise_model: Callable[[Circuit, npt.NDArray[np.floating]], Circuit],
    noise_parameters: npt.NDArray[np.floating] | Sequence[float],
    num_rounds_by_distances: Mapping[int, Sequence[int]],
    sampling_parameters: SamplingParameters = SamplingParameters(),
    memory_generator: MemoryGenerator
    | Mapping[int, Mapping[int, Circuit]] = get_rotated_surface_code_memory_circuit,
) -> ConfidenceInterval:
    """Compute 1 / Λ with an asymmetric confidence interval.

    This is the asymmetric counterpart to :func:`inverse_lambda_at`. It samples
    the same memory experiments but fits Λ with a binomial likelihood and
    propagates the asymmetric bounds into 1 / Λ. Because 1 / Λ decreases with Λ,
    the largest Λ gives the lower bound on 1 / Λ.

    Args:
        noise_model: a callable adding noise to the provided circuit, according to
            the parameters provided.
        noise_parameters: valid parameters to forward to ``noise_model``
            representing the point at which 1 / Λ should be computed.
        num_rounds_by_distances: a mapping from each code distance to the number
            of rounds used to estimate the logical error probability per round.
        sampling_parameters: additional parameters relating to the sampling tasks.
        memory_generator: a callable that generates a memory experiment.

    Returns:
        The estimate of 1 / Λ together with its lower and upper bounds.
    """
    point, noise_parameter_names, report = _run_lambda_engine(
        noise_model,
        noise_parameters,
        num_rounds_by_distances,
        sampling_parameters,
        memory_generator,
    )
    filtered = _filter_non_close_noise_parameters(
        report, point[:, 0], noise_parameter_names
    )
    lambda_data = compute_lambda_interval_from_results(
        num_rounds_by_distances, filtered
    )
    assert lambda_data.lambda_low is not None
    assert lambda_data.lambda_high is not None
    return ConfidenceInterval(
        low=1 / lambda_data.lambda_high,
        best=1 / lambda_data.lambda_,
        high=1 / lambda_data.lambda_low,
    )
