from prune import (
    get_parameter_indices_to_be_pruned_using_percentage,
    get_pruned_parameters,
)
from optimize import optimize_cost_function_with_lbfgsb
from qcbm.cost_function import get_pruned_qcbm_cost_function
import wandb
import numpy as np
from zquantum.qcbm.ansatz import QCBMAnsatz
import sys
import json

VERSION = "0.1"
PROJECT = "QLT-QCBM-find-pruning-percentage-v" + VERSION
PARAMETER_PERIOD = 2 * np.pi
USE_WANDB = True
MAX_NUMBER_OF_TRIALS = 10
if USE_WANDB:
    wandb.login()
number_of_qubits = int(sys.argv[1])
number_of_layers = int(sys.argv[2])
lbfgsb_options = {"ftol": 1e-10}

datafilename = "data/qlt/QCBM/{}/{}.json".format("L-BFGS-B", PROJECT)
try:
    with open(datafilename, "r") as f:
        DATA = json.loads(f.read())
    f.close()
except:
    DATA = {}
    with open(datafilename, "w") as f:
        f.write(json.dumps(DATA))
    f.close()
if not DATA.get(str(number_of_qubits), False):
    DATA[str(number_of_qubits)] = {}
if not DATA[str(number_of_qubits)].get(str(number_of_layers), False):
    DATA[str(number_of_qubits)][str(number_of_layers)] = []

for trial in range(MAX_NUMBER_OF_TRIALS):
    if len(DATA[str(number_of_qubits)][str(number_of_layers)]) == trial:
        DATA[str(number_of_qubits)][str(number_of_layers)].append({})
    SEED = 1234 + (number_of_layers * 17) + (trial * 23)
    np.random.seed(SEED)

    number_of_parameters = QCBMAnsatz(
        number_of_layers, number_of_qubits
    ).number_of_params
    target_distribution = np.random.uniform(0, 1, 2 ** number_of_qubits)
    target_distribution /= sum(target_distribution)
    initial_parameters = np.random.uniform(
        -1 * PARAMETER_PERIOD, PARAMETER_PERIOD, number_of_parameters
    )

    #### Unpruned Optimization
    if not DATA[str(number_of_qubits)][str(number_of_layers)][trial].get(
        "unpruned", False
    ):
        unpruned_cost_function = get_pruned_qcbm_cost_function(
            target_distribution,
            number_of_layers,
            [],
            use_wandb=USE_WANDB,
        )

        extra_config = {
            "intialization_strategy": "uniform (-{}->{})".format(
                PARAMETER_PERIOD, PARAMETER_PERIOD
            ),
            "pruning_percentage": 0,
            "parameter_period": PARAMETER_PERIOD,
            "number_of_qubits": number_of_qubits,
            "number_of_layers": number_of_layers,
            "trial": trial,
            "target_distribution": target_distribution,
            "unpruned_initial_parameters": initial_parameters,
            "pruned_parameter_indices": [],
            "number_of_pruned_parameters": 0,
            "type": "unpruned",
        }
        unpruned_results = optimize_cost_function_with_lbfgsb(
            initial_parameters,
            unpruned_cost_function,
            extra_config=extra_config,
            use_wandb=USE_WANDB,
            project=PROJECT,
            optimizer_options=lbfgsb_options,
        )
        DATA[str(number_of_qubits)][str(number_of_layers)][trial]["unpruned"] = {
            "initial_parameters": initial_parameters.tolist(),
            "seed": SEED,
            "energy": unpruned_results.opt_value,
            "optimal_parameters": unpruned_results.opt_params.tolist(),
        }
        with open(datafilename, "w") as f:
            f.write(json.dumps(DATA))
        f.close()

    for PRUNING_PERCENTAGE in [0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]:
        unpruned_optimal_parameters = np.asarray(
            DATA[str(number_of_qubits)][str(number_of_layers)][trial]["unpruned"][
                "optimal_parameters"
            ]
        )
        pruned_parameter_indices = get_parameter_indices_to_be_pruned_using_percentage(
            unpruned_optimal_parameters,
            PRUNING_PERCENTAGE,
            PARAMETER_PERIOD,
        )
        pruned_initial_parameters = get_pruned_parameters(
            initial_parameters, pruned_parameter_indices
        )

        if not DATA[str(number_of_qubits)][str(number_of_layers)][trial].get(
            str(PRUNING_PERCENTAGE), False
        ):
            DATA[str(number_of_qubits)][str(number_of_layers)][trial][
                str(PRUNING_PERCENTAGE)
            ] = {}

            pruned_cost_function = get_pruned_qcbm_cost_function(
                target_distribution,
                number_of_layers,
                pruned_parameter_indices,
                use_wandb=USE_WANDB,
            )

            extra_config = {
                "intialization_strategy": "uniform (-{}->{})".format(
                    PARAMETER_PERIOD, PARAMETER_PERIOD
                ),
                "pruning_percentage": PRUNING_PERCENTAGE,
                "parameter_period": PARAMETER_PERIOD,
                "number_of_qubits": number_of_qubits,
                "number_of_layers": number_of_layers,
                "trial": trial,
                "target_distribution": target_distribution,
                "unpruned_initial_parameters": initial_parameters,
                "pruned_parameter_indices": pruned_parameter_indices,
                "number_of_pruned_parameters": len(pruned_parameter_indices),
                "type": "pruned",
            }
            pruned_results = optimize_cost_function_with_lbfgsb(
                pruned_initial_parameters,
                pruned_cost_function,
                extra_config=extra_config,
                use_wandb=USE_WANDB,
                project=PROJECT,
                optimizer_options=lbfgsb_options,
            )
            DATA[str(number_of_qubits)][str(number_of_layers)][trial][
                str(PRUNING_PERCENTAGE)
            ]["pruned"] = {
                "initial_parameters": pruned_initial_parameters.tolist(),
                "seed": SEED,
                "energy": pruned_results.opt_value,
                "pruned_indices": pruned_parameter_indices,
                "optimal_parameters": pruned_results.opt_params.tolist(),
            }
            with open(datafilename, "w") as f:
                f.write(json.dumps(DATA))
            f.close()

        if not DATA[str(number_of_qubits)][str(number_of_layers)][trial][
            str(PRUNING_PERCENTAGE)
        ].get("pruned_and_randomized", False):
            pruned_cost_function = get_pruned_qcbm_cost_function(
                target_distribution,
                number_of_layers,
                pruned_parameter_indices,
                use_wandb=USE_WANDB,
            )
            random_initial_parameters = np.random.uniform(
                -PARAMETER_PERIOD, PARAMETER_PERIOD, len(pruned_initial_parameters)
            )

            extra_config = {
                "intialization_strategy": "uniform (-{}->{})".format(
                    PARAMETER_PERIOD, PARAMETER_PERIOD
                ),
                "pruning_percentage": PRUNING_PERCENTAGE,
                "parameter_period": PARAMETER_PERIOD,
                "number_of_qubits": number_of_qubits,
                "number_of_layers": number_of_layers,
                "trial": trial,
                "target_distribution": target_distribution,
                "unpruned_initial_parameters": initial_parameters,
                "pruned_parameter_indices": pruned_parameter_indices,
                "number_of_pruned_parameters": len(pruned_parameter_indices),
                "type": "pruned_and_randomized",
            }
            pruned_and_randomized_results = optimize_cost_function_with_lbfgsb(
                random_initial_parameters,
                pruned_cost_function,
                extra_config=extra_config,
                use_wandb=USE_WANDB,
                project=PROJECT,
                optimizer_options=lbfgsb_options,
            )
            DATA[str(number_of_qubits)][str(number_of_layers)][trial][
                str(PRUNING_PERCENTAGE)
            ]["pruned_and_randomized"] = {
                "initial_parameters": random_initial_parameters.tolist(),
                "seed": SEED,
                "energy": pruned_and_randomized_results.opt_value,
                "pruned_indices": pruned_parameter_indices,
                "optimal_parameters": pruned_and_randomized_results.opt_params.tolist(),
            }
            with open(datafilename, "w") as f:
                f.write(json.dumps(DATA))
            f.close()