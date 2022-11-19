import numpy as np
import wandb
from prune import get_padded_parameters
from unitaries import get_overparameterized_unitary


def frobenius_norm(matrix_A, matrix_B):
    return np.linalg.norm(matrix_A - matrix_B)


def get_unpruned_cost_function(
    target,
    number_of_layers,
    first_trivial_parameters,
    second_trivial_parameters,
    use_wandb=True,
):
    min_norm = np.inf

    def unpruned_cost_function(parameters):
        nonlocal min_norm
        norm = frobenius_norm(
            target, get_overparameterized_unitary(parameters, number_of_layers)
        )

        if use_wandb:
            min_norm = min(min_norm, norm)
            log_dict = {
                "Frobenius Norm": norm,
                "Minimum Frobenius Norm": min_norm,
                "Parameter Distance from First Trivial": np.linalg.norm(
                    parameters - first_trivial_parameters
                ),
                "Parameter Distance from Second Trivial": np.linalg.norm(
                    parameters - second_trivial_parameters
                ),
            }
            wandb.log(log_dict)

        return norm

    return unpruned_cost_function


def get_pruned_cost_function(
    target, number_of_layers, pruned_parameter_indices, use_wandb=True
):
    min_norm = np.inf

    def pruned_cost_function(parameters):
        nonlocal min_norm

        parameters = get_padded_parameters(parameters, pruned_parameter_indices)

        norm = frobenius_norm(
            target, get_overparameterized_unitary(parameters, number_of_layers)
        )
        if use_wandb:
            min_norm = min(min_norm, norm)
            log_dict = {
                "Frobenius Norm": norm,
                "Minimum Frobenius Norm": min_norm,
            }
            wandb.log(log_dict)

        return norm

    return pruned_cost_function
