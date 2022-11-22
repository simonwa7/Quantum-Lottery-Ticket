from optimize import (
    optimize_cost_function_with_cmaes,
)
from vqe.cost_function import get_vqe_cost_function
from vqe.hamiltonians import generate_j1j2_hamiltonian
from vqe.circuits import generate_overparameterized_vqe_j1j2_circuit
import wandb
import numpy as np
import sympy
import sys
import json
import copy

VERSION = "0.2-(popsize)"
PROJECT = "QLT-VQE-J1J2-v" + VERSION
PRUNING_CUTOFF = 1e-2
PARAMETER_PERIOD = 2 * np.pi
WEIGHT_DECAY = 50
USE_WANDB = True
MAX_NUMBER_OF_TRIALS = 30
BOUNDARY_CONDITIONS = "open"
PARAMETER_PERIOD = 2 * np.pi
J2 = 1.25
if USE_WANDB:
    wandb.login()
number_of_qubits = int(sys.argv[1])
number_of_layers = int(sys.argv[2])
popsize = int(sys.argv[3])

hamiltonian = generate_j1j2_hamiltonian(number_of_qubits, J2, j1=1)
cma_es_options = {
    "sigma_0": 0.01,
    "bounds": None,
    "tolx": 1e-10,
    "popsize": popsize,
}
SEED = 1234
np.random.seed(SEED)

number_of_parameters = (
    (3 * (2 * number_of_qubits - 3)) + number_of_qubits
) * number_of_layers
parameters = [sympy.Symbol("theta{}".format(i)) for i in range(number_of_parameters)]
parameterized_quantum_circuit = generate_overparameterized_vqe_j1j2_circuit(
    number_of_qubits, number_of_layers, parameters
)
initial_parameters = np.random.uniform(0, PARAMETER_PERIOD, number_of_parameters)

#### Unpruned Optimization
unpruned_cost_function = get_vqe_cost_function(
    hamiltonian,
    parameterized_quantum_circuit,
    pruned_indices=[],
    weight_decay=WEIGHT_DECAY,
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
    "unpruned_initial_parameters": initial_parameters,
    "pruned_parameter_indices": [],
    "number_of_pruned_parameters": 0,
    "pruning": "unpruned",
}
unpruned_results = optimize_cost_function_with_cmaes(
    initial_parameters,
    unpruned_cost_function,
    extra_config=extra_config,
    project=PROJECT,
    use_wandb=USE_WANDB,
    optimizer_options=copy.deepcopy(cma_es_options),
)
