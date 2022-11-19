from prune import get_parameter_indices_to_be_pruned, get_pruned_parameters
from optimize import optimize_cost_function_with_cmaes
from qcbm.cost_function import get_pruned_qcbm_cost_function
import wandb
import numpy as np
from zquantum.qcbm.ansatz import QCBMAnsatz

VERSION = "0.2"
PROJECT = "QLT-QCBM-v" + VERSION
PRUNING_CUTOFF = 1e-2
PARAMETER_PERIOD = 2 * np.pi
USE_WANDB = True
QUBIT_RANGE = range(2, 9, 1)
LAYER_RANGE = range(2, 10, 3)
TRIAL_RANGE = range(0, 5, 1)
if USE_WANDB:
    wandb.login()

for number_of_qubits in QUBIT_RANGE:
    for number_of_layers in LAYER_RANGE:
        for trial in TRIAL_RANGE:
            SEED = 1234 + (number_of_layers * 17) + (trial * 23)
            np.random.seed(SEED)

            number_of_parameters = QCBMAnsatz(
                number_of_layers, number_of_qubits
            ).number_of_params
            target_distribution = np.random.uniform(0, 1, 2 ** number_of_qubits)
            target_distribution /= sum(target_distribution)
            initial_parameters = np.random.uniform(
                0, PARAMETER_PERIOD, number_of_parameters
            )

            #### Unpruned Optimization
            unpruned_cost_function = get_pruned_qcbm_cost_function(
                target_distribution,
                number_of_layers,
                [],
                use_wandb=USE_WANDB,
            )

            unpruned_results = optimize_cost_function_with_cmaes(
                initial_parameters,
                unpruned_cost_function,
                PARAMETER_PERIOD,
                extra_config={
                    "intialization_strategy": "uniform (0->{})".format(
                        PARAMETER_PERIOD
                    ),
                    "target_distribution": target_distribution,
                    "prunning_cutoff": PRUNING_CUTOFF,
                    "number_of_qubits": number_of_qubits,
                    "number_of_layers": number_of_layers,
                    "trial": trial,
                    "wandb_nb": PROJECT + "-local",
                    "unpruned_initial_parameters": initial_parameters,
                    "number_of_pruned_parameters": 0,
                    "type": "unpruned",
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
            pruned_cost_function = get_pruned_qcbm_cost_function(
                target_distribution,
                number_of_layers,
                pruned_parameter_indices,
                use_wandb=USE_WANDB,
            )

            pruned_results = optimize_cost_function_with_cmaes(
                pruned_initial_parameters,
                pruned_cost_function,
                PARAMETER_PERIOD,
                extra_config={
                    "intialization_strategy": "uniform (0->{})".format(
                        PARAMETER_PERIOD
                    ),
                    "target_distribution": target_distribution,
                    "prunning_cutoff": PRUNING_CUTOFF,
                    "number_of_qubits": number_of_qubits,
                    "number_of_layers": number_of_layers,
                    "trial": trial,
                    "wandb_nb": PROJECT + "-local",
                    "unpruned_initial_parameters": initial_parameters,
                    "pruned_parameter_indices": pruned_parameter_indices,
                    "number_of_pruned_parameters": len(pruned_parameter_indices),
                    "type": "pruned",
                },
                project=PROJECT,
                use_wandb=USE_WANDB,
            )

            pruned_cost_function = get_pruned_qcbm_cost_function(
                target_distribution,
                number_of_layers,
                pruned_parameter_indices,
                use_wandb=USE_WANDB,
            )
            random_initial_parameters = np.random.uniform(
                0, PARAMETER_PERIOD, len(pruned_initial_parameters)
            )
            pruned_and_randomized_results = optimize_cost_function_with_cmaes(
                random_initial_parameters,
                pruned_cost_function,
                PARAMETER_PERIOD,
                extra_config={
                    "intialization_strategy": "uniform (0->{})".format(
                        PARAMETER_PERIOD
                    ),
                    "target_distribution": target_distribution,
                    "prunning_cutoff": PRUNING_CUTOFF,
                    "number_of_qubits": number_of_qubits,
                    "number_of_layers": number_of_layers,
                    "trial": trial,
                    "wandb_nb": PROJECT + "-local",
                    "unpruned_initial_parameters": initial_parameters,
                    "pruned_parameter_indices": pruned_parameter_indices,
                    "number_of_pruned_parameters": len(pruned_parameter_indices),
                    "type": "pruned_and_randomized",
                },
                project=PROJECT,
                use_wandb=USE_WANDB,
            )
