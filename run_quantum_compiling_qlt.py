import numpy as np
from quantum_compiling.cost_function import (
    get_pruned_cost_function,
    get_unpruned_cost_function,
)
from quantum_compiling.unitaries import get_target_unitary
from prune import get_parameter_indices_to_be_pruned, get_pruned_parameters
import wandb
from optimize import optimize_cost_function_with_lbfgsb


VERSION = "0.2"
PROJECT = "QLT-PoC-v" + VERSION
PRUNING_CUTOFF = 1e-2
PARAMETER_PERIOD = 4 * np.pi
USE_WANDB = False
LAYER_RANGE = range(1, 2, 1)
TRIAL_RANGE = range(0, 30, 1)
if USE_WANDB:
    wandb.login()


for number_of_layers in LAYER_RANGE:
    for trial in TRIAL_RANGE:
        SEED = 1234 + (number_of_layers * 17) + (trial * 23)
        np.random.seed(SEED)

        target_parameters = np.random.uniform(0, PARAMETER_PERIOD, 2)
        target = get_target_unitary(target_parameters)

        first_trivial_parameters = np.concatenate(
            (
                [target_parameters[0], 0, 0, target_parameters[1], 0, 0, 0, 0, 0, 0, 0],
                np.zeros(11 * (number_of_layers - 1)),
            )
        )
        second_trivial_parameters = np.concatenate(
            (
                [target_parameters[0], 0, 0, 0, 0, 0, 0, 0, target_parameters[1], 0, 0],
                np.zeros(11 * (number_of_layers - 1)),
            )
        )

        initial_parameters = np.random.uniform(
            -PARAMETER_PERIOD, PARAMETER_PERIOD, 11 * number_of_layers
        )

        unpruned_cost_function = get_unpruned_cost_function(
            target,
            number_of_layers,
            first_trivial_parameters,
            second_trivial_parameters,
            use_wandb=USE_WANDB,
        )
        unpruned_results = optimize_cost_function_with_lbfgsb(
            initial_parameters,
            unpruned_cost_function,
            extra_config={
                "type": "unpruned",
                "intialization_strategy": "uniform (-4pi->4pi)",
                "target_parameters": target_parameters,
                "prunning_cutoff": PRUNING_CUTOFF,
                "number_of_layers": number_of_layers,
                "trial": trial,
                "wandb_nb": PROJECT + "-local",
                "unpruned_initial_parameters": initial_parameters,
                "number_of_pruned_parameters": 0,
            },
            project=PROJECT,
            use_wandb=USE_WANDB,
        )

        pruned_parameter_indices = get_parameter_indices_to_be_pruned(
            unpruned_results.opt_params, PRUNING_CUTOFF, PARAMETER_PERIOD
        )
        pruned_initial_parameters = get_pruned_parameters(
            initial_parameters, pruned_parameter_indices
        )
        pruned_cost_function = get_pruned_cost_function(
            target, number_of_layers, pruned_parameter_indices, use_wandb=USE_WANDB
        )

        pruned_results = optimize_cost_function_with_lbfgsb(
            pruned_initial_parameters,
            pruned_cost_function,
            extra_config={
                "type": "pruned",
                "prunning_cutoff": PRUNING_CUTOFF,
                "intialization_strategy": "uniform (-4pi->4pi)",
                "target_parameters": target_parameters,
                "prunning_cutoff": PRUNING_CUTOFF,
                "number_of_layers": number_of_layers,
                "trial": trial,
                "wandb_nb": PROJECT + "-local",
                "unpruned_initial_parameters": initial_parameters,
                "pruned_parameter_indices": pruned_parameter_indices,
                "number_of_pruned_parameters": len(pruned_parameter_indices),
            },
            project=PROJECT,
            use_wandb=USE_WANDB,
        )

        pruned_cost_function = get_pruned_cost_function(
            target, number_of_layers, pruned_parameter_indices, use_wandb=USE_WANDB
        )
        random_initial_parameters = np.random.uniform(
            -PARAMETER_PERIOD, PARAMETER_PERIOD, len(pruned_initial_parameters)
        )
        pruned_and_randomized_results = optimize_cost_function_with_lbfgsb(
            random_initial_parameters,
            pruned_cost_function,
            extra_config={
                "type": "pruned_and_randomized",
                "prunning_cutoff": PRUNING_CUTOFF,
                "intialization_strategy": "uniform (-4pi->4pi)",
                "target_parameters": target_parameters,
                "prunning_cutoff": PRUNING_CUTOFF,
                "number_of_layers": number_of_layers,
                "trial": trial,
                "wandb_nb": PROJECT + "-local",
                "unpruned_initial_parameters": initial_parameters,
                "pruned_parameter_indices": pruned_parameter_indices,
                "number_of_pruned_parameters": len(pruned_parameter_indices),
            },
            project=PROJECT,
            use_wandb=USE_WANDB,
        )
