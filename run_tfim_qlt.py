from prune import get_parameter_indices_to_be_pruned, get_pruned_parameters
from optimize import optimize_cost_function_with_lbfgsb
from vqe.cost_function import get_vqe_cost_function
from vqe.hamiltonians import generate_tfim_hamiltonian
from vqe.circuits import generate_overparameterized_vqe_tfim_circuit
import wandb
from openfermion.linalg import eigenspectrum
import sympy
import numpy as np

VERSION = "0.4"
PROJECT = "QLT-Larocca-VQE-v" + VERSION
PRUNING_CUTOFF = 0.025 * (2 * np.pi)
PARAMETER_PERIOD = 2 * np.pi
USE_WANDB = False
QUBIT_RANGE = [4]
LAYER_RANGE = [50]
TRIAL_RANGE = range(1, 3, 1)
BOUNDARY_CONDITIONS = "open"
WEIGHT_DECAY = 50
if USE_WANDB:
    wandb.login()

for number_of_qubits in QUBIT_RANGE:
    print("------Running {} Qubits".format(number_of_qubits))
    hamiltonian = generate_tfim_hamiltonian(
        number_of_qubits, boundary_conditions=BOUNDARY_CONDITIONS
    )
    ground_state_energy = eigenspectrum(hamiltonian)[0]
    for number_of_layers in LAYER_RANGE:
        print("-----------Working on {} Layers".format(number_of_layers))
        number_of_parameters = 2 * number_of_layers
        parameters = [
            sympy.Symbol("theta{}".format(i)) for i in range(number_of_parameters)
        ]
        parameterized_quantum_circuit = generate_overparameterized_vqe_tfim_circuit(
            number_of_qubits, number_of_layers, parameters
        )
        for trial in TRIAL_RANGE:
            seed = (number_of_qubits * 123) + (number_of_layers * 97) + trial
            np.random.seed(seed)

            initial_parameters = np.random.uniform(
                0, PARAMETER_PERIOD, number_of_parameters
            )

            #### Unpruned Optimization
            unpruned_cost_function = get_vqe_cost_function(
                hamiltonian,
                parameterized_quantum_circuit,
                weight_decay=WEIGHT_DECAY,
                parameter_period=PARAMETER_PERIOD,
                seed=seed,
                use_wandb=USE_WANDB,
            )

            unpruned_results = optimize_cost_function_with_lbfgsb(
                initial_parameters,
                unpruned_cost_function,
                extra_config={
                    "pruning": "unpruned",
                    "hamiltonian": "tfim",
                    "boundary_conditions": BOUNDARY_CONDITIONS,
                    "prunning_cutoff": PRUNING_CUTOFF,
                    "ground_state_energy": ground_state_energy,
                    "intialization_strategy": "uniform (0->2pi)",
                    "number_of_layers": number_of_layers,
                    "number_of_qubits": number_of_qubits,
                    "trial": trial,
                    "weight decay": WEIGHT_DECAY,
                    "parameter_period": PARAMETER_PERIOD,
                    "seed": seed,
                    "unpruned_initial_parameters": initial_parameters,
                    "number_of_pruned_parameters": 0,
                },
                project=PROJECT,
                use_wandb=USE_WANDB,
                optimizer_options={"ftol": 1e-10},
            )

            pruned_parameter_indices = get_parameter_indices_to_be_pruned(
                unpruned_results.opt_params, PRUNING_CUTOFF, PARAMETER_PERIOD
            )

            pruned_initial_parameters = get_pruned_parameters(
                initial_parameters, pruned_parameter_indices
            )
            print(
                "---------------Trial {} Resulting in {} Pruned Parameters".format(
                    trial, len(pruned_parameter_indices)
                )
            )
            pruned_cost_function = get_vqe_cost_function(
                hamiltonian,
                parameterized_quantum_circuit,
                pruned_indices=pruned_parameter_indices,
                weight_decay=WEIGHT_DECAY,
                parameter_period=PARAMETER_PERIOD,
                seed=seed,
                use_wandb=USE_WANDB,
            )

            pruned_results = optimize_cost_function_with_lbfgsb(
                pruned_initial_parameters,
                pruned_cost_function,
                extra_config={
                    "pruning": "pruned",
                    "hamiltonian": "tfim",
                    "boundary_conditions": BOUNDARY_CONDITIONS,
                    "prunning_cutoff": PRUNING_CUTOFF,
                    "ground_state_energy": ground_state_energy,
                    "intialization_strategy": "uniform (0->2pi)",
                    "number_of_layers": number_of_layers,
                    "number_of_qubits": number_of_qubits,
                    "weight decay": WEIGHT_DECAY,
                    "parameter_period": PARAMETER_PERIOD,
                    "trial": trial,
                    "seed": seed,
                    "unpruned_initial_parameters": initial_parameters,
                    "pruned_parameter_indices": pruned_parameter_indices,
                    "number_of_pruned_parameters": len(pruned_parameter_indices),
                },
                project=PROJECT,
                use_wandb=USE_WANDB,
                optimizer_options={"ftol": 1e-10},
            )

            pruned_cost_function = get_vqe_cost_function(
                hamiltonian,
                parameterized_quantum_circuit,
                pruned_indices=pruned_parameter_indices,
                weight_decay=WEIGHT_DECAY,
                parameter_period=PARAMETER_PERIOD,
                seed=seed,
                use_wandb=USE_WANDB,
            )
            random_initial_parameters = np.random.uniform(
                -1 * PARAMETER_PERIOD, PARAMETER_PERIOD, len(pruned_initial_parameters)
            )

            pruned_and_randomized_results = optimize_cost_function_with_lbfgsb(
                random_initial_parameters,
                pruned_cost_function,
                extra_config={
                    "pruning": "pruned_and_randomized",
                    "hamiltonian": "tfim",
                    "boundary_conditions": BOUNDARY_CONDITIONS,
                    "prunning_cutoff": PRUNING_CUTOFF,
                    "ground_state_energy": ground_state_energy,
                    "intialization_strategy": "uniform (0->2pi)",
                    "number_of_layers": number_of_layers,
                    "number_of_qubits": number_of_qubits,
                    "weight decay": WEIGHT_DECAY,
                    "parameter_period": PARAMETER_PERIOD,
                    "trial": trial,
                    "seed": seed,
                    "unpruned_initial_parameters": initial_parameters,
                    "pruned_parameter_indices": pruned_parameter_indices,
                    "number_of_pruned_parameters": len(pruned_parameter_indices),
                },
                project=PROJECT,
                use_wandb=USE_WANDB,
                optimizer_options={"ftol": 1e-10},
            )
