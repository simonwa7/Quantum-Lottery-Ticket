import wandb
import numpy as np
from zquantum.core.cost_function import get_ground_state_cost_function
from zquantum.core.estimation import calculate_exact_expectation_values
from qecirq.simulator import CirqSimulator
from prune import get_padded_parameters


def get_vqe_cost_function(
    hamiltonian,
    parameterized_quantum_circuit,
    pruned_indices=[],
    weight_decay=0,
    parameter_period=2 * np.pi,
    seed=123,
    use_wandb=False,
):
    cost_function = get_ground_state_cost_function(
        hamiltonian,
        parameterized_quantum_circuit,
        CirqSimulator(seed=seed),
        estimation_method=calculate_exact_expectation_values,
    )

    min_energy = np.inf
    min_cost = np.inf

    def wrapped_cost_function(parameters):
        nonlocal min_energy, min_cost
        parameters = get_padded_parameters(parameters, pruned_indices)
        energy = cost_function(parameters)

        # Add bias to drive parameters to zero-values
        max_parameter_distances = np.asarray([np.pi for _ in parameters])
        parameter_distances = np.asarray(
            [
                min(
                    parameter % (parameter_period),
                    np.abs((parameter % parameter_period) - parameter_period),
                )
                for parameter in parameters
            ]
        )
        bias = weight_decay * (
            sum(parameter_distances ** 2) / sum(max_parameter_distances ** 2)
        )
        cost = energy + bias

        if use_wandb:
            min_energy = min(min_energy, energy)
            min_cost = min(min_cost, cost)
            log_dict = {
                "Energy": energy,
                "Minimum Energy": min_energy,
                "Parameter Weight Bias": bias,
                "Cost": cost,
                "Minimum Cost": min_cost,
            }
            wandb.log(log_dict)

        return cost

    return wrapped_cost_function
