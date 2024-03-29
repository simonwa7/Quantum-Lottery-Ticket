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

VERSION = "0.15"
PROJECT = "QLT-QCBM-v" + VERSION
PRUNING_PERCENTAGE = 0.9
PARAMETER_PERIOD = 2 * np.pi
WEIGHT_DECAY = 0
USE_WANDB = True
MAX_NUMBER_OF_TRIALS = 10
DISTRIBUTION_TYPE = "normal"
DISTRIBUTION_MEAN = 0.65
DISTRIBUTION_STDEV = 0.1
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
    SEED = 1234 + (number_of_layers * 17) + (trial * 23) + (abs(hash(PROJECT)) % 10000)
    np.random.seed(SEED)

    number_of_parameters = QCBMAnsatz(
        number_of_layers, number_of_qubits
    ).number_of_params

    #### Unpruned Optimization
    if not DATA[str(number_of_qubits)][str(number_of_layers)][trial].get(
        "unpruned", False
    ):
        samples = np.random.normal(
            loc=DISTRIBUTION_MEAN, scale=DISTRIBUTION_STDEV, size=100000000
        )
        target_distribution = np.histogram(samples, 2 ** number_of_qubits, (0, 1))[0]
        target_distribution = target_distribution / sum(target_distribution)
        initial_parameters = np.random.uniform(
            (-1 * PARAMETER_PERIOD) / 2, (PARAMETER_PERIOD) / 2, number_of_parameters
        )

        unpruned_cost_function = get_pruned_qcbm_cost_function(
            target_distribution,
            number_of_layers,
            [],
            use_wandb=USE_WANDB,
            weight_decay=WEIGHT_DECAY,
            parameter_period=PARAMETER_PERIOD,
        )

        extra_config = {
            "intialization_strategy": "uniform (-{}->{})".format(
                PARAMETER_PERIOD / 2, PARAMETER_PERIOD / 2
            ),
            "pruning_percentage": 0,
            "weight_decay": WEIGHT_DECAY,
            "parameter_period": PARAMETER_PERIOD,
            "number_of_qubits": number_of_qubits,
            "number_of_layers": number_of_layers,
            "trial": trial,
            "target_distribution": target_distribution,
            "unpruned_initial_parameters": initial_parameters,
            "initial_parameters": initial_parameters,
            "pruned_parameter_indices": [],
            "number_of_pruned_parameters": 0,
            "type": "unpruned",
            "target_distribution_type": DISTRIBUTION_TYPE,
            "target_distribution_mean": DISTRIBUTION_MEAN,
            "target_distribution_stddev": DISTRIBUTION_STDEV,
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
            "target_distribution": target_distribution.tolist(),
            "seed": SEED,
            "energy": unpruned_results.opt_value,
            "optimal_parameters": unpruned_results.opt_params.tolist(),
        }
        with open(datafilename, "w") as f:
            f.write(json.dumps(DATA))
        f.close()

    initial_parameters = np.asarray(
        DATA[str(number_of_qubits)][str(number_of_layers)][trial]["unpruned"][
            "initial_parameters"
        ]
    )
    target_distribution = np.asarray(
        DATA[str(number_of_qubits)][str(number_of_layers)][trial]["unpruned"][
            "target_distribution"
        ]
    )
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
        "pruned:{}".format(PRUNING_PERCENTAGE), False
    ):
        pruned_cost_function = get_pruned_qcbm_cost_function(
            target_distribution,
            number_of_layers,
            pruned_parameter_indices,
            use_wandb=USE_WANDB,
            weight_decay=WEIGHT_DECAY,
            parameter_period=PARAMETER_PERIOD,
        )

        extra_config = {
            "intialization_strategy": "uniform (-{}->{})".format(
                PARAMETER_PERIOD / 2, PARAMETER_PERIOD / 2
            ),
            "pruning_percentage": PRUNING_PERCENTAGE,
            "weight_decay": WEIGHT_DECAY,
            "parameter_period": PARAMETER_PERIOD,
            "number_of_qubits": number_of_qubits,
            "number_of_layers": number_of_layers,
            "trial": trial,
            "target_distribution": target_distribution,
            "unpruned_initial_parameters": initial_parameters,
            "initial_parameters": pruned_initial_parameters,
            "pruned_parameter_indices": pruned_parameter_indices,
            "number_of_pruned_parameters": len(pruned_parameter_indices),
            "type": "pruned",
            "target_distribution_type": DISTRIBUTION_TYPE,
            "target_distribution_mean": DISTRIBUTION_MEAN,
            "target_distribution_stddev": DISTRIBUTION_STDEV,
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
            "pruned:{}".format(PRUNING_PERCENTAGE)
        ] = {
            "initial_parameters": pruned_initial_parameters.tolist(),
            "seed": SEED,
            "energy": pruned_results.opt_value,
            "pruned_indices": pruned_parameter_indices,
            "optimal_parameters": pruned_results.opt_params.tolist(),
        }
        with open(datafilename, "w") as f:
            f.write(json.dumps(DATA))
        f.close()

    for PARAMETER_DISPLACEMENT in [
        # (1 / 1024) * PARAMETER_PERIOD,
        # (1 / 512) * PARAMETER_PERIOD,
        # (1 / 256) * PARAMETER_PERIOD,
        (1 / 128) * PARAMETER_PERIOD,
        # (1 / 64) * PARAMETER_PERIOD,
        (1 / 32) * PARAMETER_PERIOD,
        (1 / 8) * PARAMETER_PERIOD,
        (1 / 2) * PARAMETER_PERIOD,
    ]:
        if not DATA[str(number_of_qubits)][str(number_of_layers)][trial].get(
            "pruned_with_displacement:{}|{}".format(
                PRUNING_PERCENTAGE, PARAMETER_DISPLACEMENT
            ),
            False,
        ):
            pruned_cost_function = get_pruned_qcbm_cost_function(
                target_distribution,
                number_of_layers,
                pruned_parameter_indices,
                use_wandb=USE_WANDB,
                weight_decay=WEIGHT_DECAY,
                parameter_period=PARAMETER_PERIOD,
            )

            pruned_and_displaced_initial_parameters = (
                pruned_initial_parameters
                + np.random.uniform(
                    0, PARAMETER_DISPLACEMENT, len(pruned_initial_parameters)
                )
            )

            extra_config = {
                "intialization_strategy": "uniform (-{}->{})".format(
                    PARAMETER_PERIOD / 2, PARAMETER_PERIOD / 2
                ),
                "pruning_percentage": PRUNING_PERCENTAGE,
                "parameter_displacement": PARAMETER_DISPLACEMENT,
                "weight_decay": WEIGHT_DECAY,
                "parameter_period": PARAMETER_PERIOD,
                "number_of_qubits": number_of_qubits,
                "number_of_layers": number_of_layers,
                "trial": trial,
                "target_distribution": target_distribution,
                "unpruned_initial_parameters": initial_parameters,
                "initial_parameters": pruned_and_displaced_initial_parameters,
                "pruned_parameter_indices": pruned_parameter_indices,
                "number_of_pruned_parameters": len(pruned_parameter_indices),
                "type": "pruned_with_displacement:{}".format(PARAMETER_DISPLACEMENT),
                "target_distribution_type": DISTRIBUTION_TYPE,
                "target_distribution_mean": DISTRIBUTION_MEAN,
                "target_distribution_stddev": DISTRIBUTION_STDEV,
            }
            pruned_results = optimize_cost_function_with_lbfgsb(
                pruned_and_displaced_initial_parameters,
                pruned_cost_function,
                extra_config=extra_config,
                use_wandb=USE_WANDB,
                project=PROJECT,
                optimizer_options=lbfgsb_options,
            )
            DATA[str(number_of_qubits)][str(number_of_layers)][trial][
                "pruned_with_displacement:{}|{}".format(
                    PRUNING_PERCENTAGE, PARAMETER_DISPLACEMENT
                )
            ] = {
                "initial_parameters": pruned_and_displaced_initial_parameters.tolist(),
                "seed": SEED,
                "energy": pruned_results.opt_value,
                "pruned_indices": pruned_parameter_indices,
                "optimal_parameters": pruned_results.opt_params.tolist(),
            }
            with open(datafilename, "w") as f:
                f.write(json.dumps(DATA))
            f.close()

    if not DATA[str(number_of_qubits)][str(number_of_layers)][trial].get(
        "pruned_and_randomized:{}".format(PRUNING_PERCENTAGE), False
    ):
        pruned_cost_function = get_pruned_qcbm_cost_function(
            target_distribution,
            number_of_layers,
            pruned_parameter_indices,
            use_wandb=USE_WANDB,
            weight_decay=WEIGHT_DECAY,
            parameter_period=PARAMETER_PERIOD,
        )
        random_initial_parameters = np.random.uniform(
            (-PARAMETER_PERIOD) / 2,
            (PARAMETER_PERIOD) / 2,
            len(pruned_initial_parameters),
        )

        extra_config = {
            "intialization_strategy": "uniform (-{}->{})".format(
                PARAMETER_PERIOD / 2, PARAMETER_PERIOD / 2
            ),
            "pruning_percentage": PRUNING_PERCENTAGE,
            "weight_decay": WEIGHT_DECAY,
            "parameter_period": PARAMETER_PERIOD,
            "number_of_qubits": number_of_qubits,
            "number_of_layers": number_of_layers,
            "trial": trial,
            "target_distribution": target_distribution,
            "unpruned_initial_parameters": initial_parameters,
            "initial_parameters": random_initial_parameters,
            "pruned_parameter_indices": pruned_parameter_indices,
            "number_of_pruned_parameters": len(pruned_parameter_indices),
            "type": "pruned_and_randomized",
            "target_distribution_type": DISTRIBUTION_TYPE,
            "target_distribution_mean": DISTRIBUTION_MEAN,
            "target_distribution_stddev": DISTRIBUTION_STDEV,
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
            "pruned_and_randomized:{}".format(PRUNING_PERCENTAGE)
        ] = {
            "initial_parameters": random_initial_parameters.tolist(),
            "seed": SEED,
            "energy": pruned_and_randomized_results.opt_value,
            "pruned_indices": pruned_parameter_indices,
            "optimal_parameters": pruned_and_randomized_results.opt_params.tolist(),
        }
        with open(datafilename, "w") as f:
            f.write(json.dumps(DATA))
        f.close()

    if not DATA[str(number_of_qubits)][str(number_of_layers)][trial].get(
        "random_subnetwork:{}".format(PRUNING_PERCENTAGE), False
    ):
        randomly_pruned_indices = [
            int(index)
            for index in sorted(
                np.random.choice(
                    range(number_of_parameters),
                    len(pruned_parameter_indices),
                    replace=False,
                )
            )
        ]

        randomly_pruned_initial_parameters = get_pruned_parameters(
            initial_parameters, randomly_pruned_indices
        )

        pruned_cost_function = get_pruned_qcbm_cost_function(
            target_distribution,
            number_of_layers,
            randomly_pruned_indices,
            use_wandb=USE_WANDB,
            weight_decay=WEIGHT_DECAY,
            parameter_period=PARAMETER_PERIOD,
        )

        extra_config = {
            "intialization_strategy": "uniform (-{}->{})".format(
                PARAMETER_PERIOD / 2, PARAMETER_PERIOD / 2
            ),
            "pruning_percentage": PRUNING_PERCENTAGE,
            "weight_decay": WEIGHT_DECAY,
            "parameter_period": PARAMETER_PERIOD,
            "number_of_qubits": number_of_qubits,
            "number_of_layers": number_of_layers,
            "trial": trial,
            "target_distribution": target_distribution,
            "unpruned_initial_parameters": initial_parameters,
            "initial_parameters": randomly_pruned_initial_parameters,
            "pruned_parameter_indices": randomly_pruned_indices,
            "number_of_pruned_parameters": len(randomly_pruned_indices),
            "type": "random_subnetwork",
            "target_distribution_type": DISTRIBUTION_TYPE,
            "target_distribution_mean": DISTRIBUTION_MEAN,
            "target_distribution_stddev": DISTRIBUTION_STDEV,
        }
        random_subnetwork_results = optimize_cost_function_with_lbfgsb(
            randomly_pruned_initial_parameters,
            pruned_cost_function,
            extra_config=extra_config,
            use_wandb=USE_WANDB,
            project=PROJECT,
            optimizer_options=lbfgsb_options,
        )
        DATA[str(number_of_qubits)][str(number_of_layers)][trial][
            "random_subnetwork:{}".format(PRUNING_PERCENTAGE)
        ] = {
            "initial_parameters": randomly_pruned_initial_parameters.tolist(),
            "seed": SEED,
            "energy": random_subnetwork_results.opt_value,
            "pruned_indices": randomly_pruned_indices,
            "optimal_parameters": random_subnetwork_results.opt_params.tolist(),
        }
        with open(datafilename, "w") as f:
            f.write(json.dumps(DATA))
        f.close()

    randomly_pruned_indices = DATA[str(number_of_qubits)][str(number_of_layers)][trial][
        "random_subnetwork:{}".format(PRUNING_PERCENTAGE)
    ]["pruned_indices"]

    if not DATA[str(number_of_qubits)][str(number_of_layers)][trial].get(
        "random_subnetwork_randomized_parameters:{}".format(PRUNING_PERCENTAGE), False
    ):
        random_initial_parameters = np.random.uniform(
            (-PARAMETER_PERIOD) / 2,
            (PARAMETER_PERIOD) / 2,
            number_of_parameters - len(randomly_pruned_indices),
        )

        pruned_cost_function = get_pruned_qcbm_cost_function(
            target_distribution,
            number_of_layers,
            randomly_pruned_indices,
            use_wandb=USE_WANDB,
            weight_decay=WEIGHT_DECAY,
            parameter_period=PARAMETER_PERIOD,
        )

        extra_config = {
            "intialization_strategy": "uniform (-{}->{})".format(
                PARAMETER_PERIOD / 2, PARAMETER_PERIOD / 2
            ),
            "pruning_percentage": PRUNING_PERCENTAGE,
            "weight_decay": WEIGHT_DECAY,
            "parameter_period": PARAMETER_PERIOD,
            "number_of_qubits": number_of_qubits,
            "number_of_layers": number_of_layers,
            "trial": trial,
            "target_distribution": target_distribution,
            "unpruned_initial_parameters": None,
            "initial_parameters": random_initial_parameters,
            "pruned_parameter_indices": randomly_pruned_indices,
            "number_of_pruned_parameters": len(randomly_pruned_indices),
            "type": "random_subnetwork_randomized_parameters",
            "target_distribution_type": DISTRIBUTION_TYPE,
            "target_distribution_mean": DISTRIBUTION_MEAN,
            "target_distribution_stddev": DISTRIBUTION_STDEV,
        }
        random_subnetwork_randomized_parameters_results = (
            optimize_cost_function_with_lbfgsb(
                random_initial_parameters,
                pruned_cost_function,
                extra_config=extra_config,
                use_wandb=USE_WANDB,
                project=PROJECT,
                optimizer_options=lbfgsb_options,
            )
        )
        DATA[str(number_of_qubits)][str(number_of_layers)][trial][
            "random_subnetwork_randomized_parameters:{}".format(PRUNING_PERCENTAGE)
        ] = {
            "initial_parameters": random_initial_parameters.tolist(),
            "seed": SEED,
            "energy": random_subnetwork_randomized_parameters_results.opt_value,
            "pruned_indices": randomly_pruned_indices,
            "optimal_parameters": random_subnetwork_randomized_parameters_results.opt_params.tolist(),
        }
        with open(datafilename, "w") as f:
            f.write(json.dumps(DATA))
        f.close()
