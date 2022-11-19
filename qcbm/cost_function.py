from ..prune import get_padded_parameters
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

    def pruned_cost_function(parameters):
        nonlocal min_cost

        parameters = get_padded_parameters(parameters, pruned_indices)

        cost = unpruned_cost_function(parameters)
        min_cost = min(min_cost, cost)
        if use_wandb:
            log_dict = {
                "Clipped Negative Log Likelihood": cost,
                "Minimum Clipped Negative Log Likelihood": min_cost,
                "Offset Clipped Negative Log Likelihood": cost - minimum_possible_cnll,
                "Minimum Offset Clipped Negative Log Likelihood": min_cost
                - minimum_possible_cnll,
                "Number of Circuits Run": backend.number_of_circuits_run,
            }
            wandb.log(log_dict)

        return cost

    return pruned_cost_function
