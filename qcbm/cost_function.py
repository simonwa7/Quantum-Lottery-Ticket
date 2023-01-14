from prune import get_padded_parameters, calculate_parameter_weight_bias
from zquantum.qcbm.ansatz import QCBMAnsatz
from zquantum.qcbm.cost_function import create_QCBM_cost_function
from qeqiskit.simulator import QiskitSimulator
from zquantum.core.distribution import (
    compute_clipped_negative_log_likelihood,
    create_bitstring_distribution_from_probability_distribution,
)
import numpy as np
import wandb


def get_pruned_qcbm_cost_function(
    target_distribution,
    number_of_layers,
    pruned_indices,
    use_wandb=True,
    n_samples=None,
    parameter_period=2 * np.pi,
    weight_decay=0,
):
    number_of_qubits = int(np.log2(len(target_distribution)))
    target_distribution = create_bitstring_distribution_from_probability_distribution(
        target_distribution
    )
    ansatz = QCBMAnsatz(number_of_layers, number_of_qubits)
    backend = QiskitSimulator("statevector_simulator")

    distance_measure = compute_clipped_negative_log_likelihood
    minimum_possible_cnll = compute_clipped_negative_log_likelihood(
        target_distribution, target_distribution, {}
    )
    distance_measure_parameters = {}
    unpruned_cost_function = create_QCBM_cost_function(
        ansatz=ansatz,
        backend=backend,
        n_samples=n_samples,
        distance_measure=distance_measure,
        distance_measure_parameters=distance_measure_parameters,
        target_bitstring_distribution=target_distribution,
    )

    min_cost = np.inf
    min_cnll = np.inf

    def pruned_cost_function(parameters):
        nonlocal min_cost, min_cnll

        parameters = get_padded_parameters(parameters, pruned_indices)

        cnll = unpruned_cost_function(parameters)
        parameter_bias = calculate_parameter_weight_bias(
            parameters, parameter_period, weight_decay
        )
        cost = cnll + parameter_bias
        min_cost = min(min_cost, cost)
        min_cnll = min(min_cnll, cnll)
        if use_wandb:
            log_dict = {
                "Minimum Offset Cost": min_cost - minimum_possible_cnll,
                "Minimum Offset Clipped Negative Log Likelihood": min_cnll
                - minimum_possible_cnll,
                "Offset Cost": cost - minimum_possible_cnll,
                "Offset Clipped Negative Log Likelihood": cnll - minimum_possible_cnll,
                "Cost": cost,
                "Minimum Cost": min_cost,
                "Parameter Weight Bias": parameter_bias,
                "Clipped Negative Log Likelihood": cnll,
                "Minimum Clipped Negative Log Likelihood": min_cnll,
                "Number of Circuits Run": backend.number_of_circuits_run,
            }
            wandb.log(log_dict)

        return cost

    return pruned_cost_function
