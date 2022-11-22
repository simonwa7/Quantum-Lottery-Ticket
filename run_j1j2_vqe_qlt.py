from prune import get_parameter_indices_to_be_pruned, get_pruned_parameters
from optimize import (
    optimize_cost_function_with_cmaes,
    optimize_cost_function_with_lbfgsb,
)
from vqe.cost_function import get_vqe_cost_function
from vqe.hamiltonians import generate_j1j2_hamiltonian
from vqe.circuits import generate_overparameterized_vqe_j1j2_circuit
from openfermion.linalg import eigenspectrum
import wandb
import numpy as np
import sympy
import sys
import json
import copy

VERSION = "0.1"
PROJECT = "QLT-VQE-J1J2-v" + VERSION
PRUNING_CUTOFF = 1e-2
PARAMETER_PERIOD = 2 * np.pi
WEIGHT_DECAY = 50
USE_WANDB = False
MAX_NUMBER_OF_TRIALS = 30
BOUNDARY_CONDITIONS = "open"
PARAMETER_PERIOD = 2 * np.pi
J2 = 1.25
if USE_WANDB:
    wandb.login()
number_of_qubits = int(sys.argv[1])
number_of_layers = int(sys.argv[2])
optimizer = str(sys.argv[3])
lbfgsb_options = {"ftol": 1e-10}
cma_es_options = {
    "sigma_0": 0.01,
    "bounds": None,
    "tolx": 1e-10,
    "popsize": 36,
    "maxfevals": 20000,
}

datafilename = "data/qlt/j1j2_{}.json".format(optimizer)
with open(datafilename, "r") as f:
    DATA = json.loads(f.read())
f.close()
if not DATA.get(str(number_of_qubits), False):
    DATA[str(number_of_qubits)] = {}
if not DATA[str(number_of_qubits)].get(str(number_of_layers), False):
    DATA[str(number_of_qubits)][str(number_of_layers)] = []

hamiltonian = generate_j1j2_hamiltonian(number_of_qubits, J2, j1=1)
ground_state_energy = eigenspectrum(hamiltonian)[0]

for trial in range(MAX_NUMBER_OF_TRIALS):
    if len(DATA[str(number_of_qubits)][str(number_of_layers)]) == trial:
        DATA[str(number_of_qubits)][str(number_of_layers)].append({})
    SEED = 1234 + (number_of_layers * 17) + (trial * 23)
    np.random.seed(SEED)

    number_of_parameters = (
        (3 * (2 * number_of_qubits - 3)) + number_of_qubits
    ) * number_of_layers
    parameters = [
        sympy.Symbol("theta{}".format(i)) for i in range(number_of_parameters)
    ]
    parameterized_quantum_circuit = generate_overparameterized_vqe_j1j2_circuit(
        number_of_qubits, number_of_layers, parameters
    )
    initial_parameters = np.random.uniform(0, PARAMETER_PERIOD, number_of_parameters)

    #### Unpruned Optimization
    if not DATA[str(number_of_qubits)][str(number_of_layers)][trial].get(
        "unpruned", False
    ):
        unpruned_cost_function = get_vqe_cost_function(
            hamiltonian,
            parameterized_quantum_circuit,
            pruned_indices=[],
            weight_decay=WEIGHT_DECAY,
            offset=-1 * ground_state_energy,
            parameter_period=PARAMETER_PERIOD,
            seed=SEED,
            use_wandb=USE_WANDB,
        )

        extra_config = {
            "intialization_strategy": "uniform (0->{})".format(PARAMETER_PERIOD),
            "prunning_cutoff": PRUNING_CUTOFF,
            "boundary_conditions": BOUNDARY_CONDITIONS,
            "weight decay": WEIGHT_DECAY,
            "parameter_period": PARAMETER_PERIOD,
            "J2": J2,
            "number_of_qubits": number_of_qubits,
            "number_of_layers": number_of_layers,
            "trial": trial,
            "unpruned_initial_parameters": initial_parameters,
            "pruned_parameter_indices": [],
            "number_of_pruned_parameters": 0,
            "pruning": "unpruned",
        }
        if optimizer == "L-BFGS-B":
            unpruned_results = optimize_cost_function_with_lbfgsb(
                initial_parameters,
                unpruned_cost_function,
                extra_config=extra_config,
                project=PROJECT,
                use_wandb=USE_WANDB,
                optimizer_options=lbfgsb_options,
            )
        elif optimizer == "CMA-ES":
            unpruned_results = optimize_cost_function_with_cmaes(
                initial_parameters,
                unpruned_cost_function,
                extra_config=extra_config,
                project=PROJECT,
                use_wandb=USE_WANDB,
                optimizer_options=copy.deepcopy(cma_es_options),
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

    if not DATA[str(number_of_qubits)][str(number_of_layers)][trial].get(
        "pruned", False
    ):
        assert np.array_equal(
            initial_parameters,
            np.asarray(
                DATA[str(number_of_qubits)][str(number_of_layers)][trial]["unpruned"][
                    "initial_parameters"
                ]
            ),
        )
        unpruned_optimal_parameters = np.asarray(
            DATA[str(number_of_qubits)][str(number_of_layers)][trial]["unpruned"][
                "optimal_parameters"
            ]
        )
        pruned_parameter_indices = get_parameter_indices_to_be_pruned(
            unpruned_optimal_parameters,
            PRUNING_CUTOFF,
            PARAMETER_PERIOD,
        )

        pruned_initial_parameters = get_pruned_parameters(
            initial_parameters, pruned_parameter_indices
        )

        pruned_cost_function = get_vqe_cost_function(
            hamiltonian,
            parameterized_quantum_circuit,
            pruned_indices=pruned_parameter_indices,
            weight_decay=WEIGHT_DECAY,
            offset=-1 * ground_state_energy,
            parameter_period=PARAMETER_PERIOD,
            seed=SEED,
            use_wandb=USE_WANDB,
        )

        extra_config = {
            "intialization_strategy": "uniform (0->{})".format(PARAMETER_PERIOD),
            "prunning_cutoff": PRUNING_CUTOFF,
            "boundary_conditions": BOUNDARY_CONDITIONS,
            "weight decay": WEIGHT_DECAY,
            "parameter_period": PARAMETER_PERIOD,
            "J2": J2,
            "number_of_qubits": number_of_qubits,
            "number_of_layers": number_of_layers,
            "trial": trial,
            "unpruned_initial_parameters": initial_parameters,
            "pruned_parameter_indices": pruned_parameter_indices,
            "number_of_pruned_parameters": len(pruned_parameter_indices),
            "pruning": "pruned",
        }
        if optimizer == "L-BFGS-B":
            pruned_results = optimize_cost_function_with_lbfgsb(
                pruned_initial_parameters,
                pruned_cost_function,
                extra_config=extra_config,
                project=PROJECT,
                use_wandb=USE_WANDB,
                optimizer_options=lbfgsb_options,
            )
        elif optimizer == "CMA-ES":
            pruned_results = optimize_cost_function_with_cmaes(
                pruned_initial_parameters,
                pruned_cost_function,
                extra_config=extra_config,
                project=PROJECT,
                use_wandb=USE_WANDB,
                optimizer_options=copy.deepcopy(cma_es_options),
            )

        DATA[str(number_of_qubits)][str(number_of_layers)][trial]["pruned"] = {
            "initial_parameters": pruned_initial_parameters.tolist(),
            "seed": SEED,
            "energy": pruned_results.opt_value,
            "pruned_indices": pruned_parameter_indices,
            "optimal_parameters": pruned_results.opt_params.tolist(),
        }
        with open(datafilename, "w") as f:
            f.write(json.dumps(DATA))
        f.close()

    if not DATA[str(number_of_qubits)][str(number_of_layers)][trial].get(
        "pruned_and_randomized", False
    ):
        pruned_cost_function = get_vqe_cost_function(
            hamiltonian,
            parameterized_quantum_circuit,
            pruned_indices=pruned_parameter_indices,
            weight_decay=WEIGHT_DECAY,
            offset=-1 * ground_state_energy,
            parameter_period=PARAMETER_PERIOD,
            seed=SEED,
            use_wandb=USE_WANDB,
        )
        random_initial_parameters = np.random.uniform(
            0, PARAMETER_PERIOD, len(pruned_initial_parameters)
        )

        extra_config = {
            "intialization_strategy": "uniform (0->{})".format(PARAMETER_PERIOD),
            "prunning_cutoff": PRUNING_CUTOFF,
            "boundary_conditions": BOUNDARY_CONDITIONS,
            "weight decay": WEIGHT_DECAY,
            "parameter_period": PARAMETER_PERIOD,
            "J2": J2,
            "number_of_qubits": number_of_qubits,
            "number_of_layers": number_of_layers,
            "trial": trial,
            "unpruned_initial_parameters": initial_parameters,
            "pruned_parameter_indices": pruned_parameter_indices,
            "number_of_pruned_parameters": len(pruned_parameter_indices),
            "pruning": "pruned_and_randomized",
        }
        if optimizer == "L-BFGS-B":
            pruned_and_randomized_results = optimize_cost_function_with_lbfgsb(
                random_initial_parameters,
                pruned_cost_function,
                extra_config=extra_config,
                project=PROJECT,
                use_wandb=USE_WANDB,
                optimizer_options=lbfgsb_options,
            )
        elif optimizer == "CMA-ES":
            pruned_and_randomized_results = optimize_cost_function_with_cmaes(
                random_initial_parameters,
                pruned_cost_function,
                extra_config=extra_config,
                project=PROJECT,
                use_wandb=USE_WANDB,
                optimizer_options=copy.deepcopy(cma_es_options),
            )

        DATA[str(number_of_qubits)][str(number_of_layers)][trial][
            "pruned_and_randomized"
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
