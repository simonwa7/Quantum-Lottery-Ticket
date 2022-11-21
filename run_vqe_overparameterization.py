from vqe.cost_function import get_vqe_cost_function
from vqe.circuits import (
    generate_alternating_vqe_j1j2_circuit,
    generate_overparameterized_vqe_tfim_circuit,
    generate_overparameterized_vqe_j1j2_circuit,
)
from vqe.hamiltonians import generate_tfim_hamiltonian, generate_j1j2_hamiltonian
from optimize import (
    optimize_cost_function_with_lbfgsb,
    optimize_cost_function_with_cmaes,
)
from openfermion.linalg import eigenspectrum
import sympy
import numpy as np
import json
import sys

RECORD_DATA = True
TRIAL_RANGE = range(10)
BOUNDARY_CONDITIONS = "open"
CIRCUIT_TYPE = "j1j2_alternating-ansatz"
PARAMETER_PERIOD = 2 * np.pi
J2 = 1.25
OPTIMIZER = "CMA-ES"

datafilename = "data/overparameterization_{}_data.json".format(CIRCUIT_TYPE)
with open(datafilename, "r") as f:
    DATA = json.loads(f.read())
f.close()

number_of_qubits = int(sys.argv[1])
number_of_layers = int(sys.argv[2])

##### GENERATE HAMILTONIAN #####
qubit_data = DATA.get(str(number_of_qubits), {})
DATA[str(number_of_qubits)] = qubit_data

if CIRCUIT_TYPE == "tfim":
    hamiltonian = generate_tfim_hamiltonian(
        number_of_qubits, boundary_conditions=BOUNDARY_CONDITIONS
    )
elif (CIRCUIT_TYPE == "j1j2") or (CIRCUIT_TYPE == "j1j2_alternating-ansatz"):
    hamiltonian = generate_j1j2_hamiltonian(number_of_qubits, J2, j1=1)

ground_state_energy = eigenspectrum(hamiltonian)[0]
DATA[str(number_of_qubits)]["ground_state_energy"] = ground_state_energy


##### GENERATE PARAMETERIZED QUANTUM CIRCUIT #####
if CIRCUIT_TYPE == "tfim":
    number_of_parameters = 2 * number_of_layers
    parameters = [
        sympy.Symbol("theta{}".format(i)) for i in range(number_of_parameters)
    ]
    parameterized_quantum_circuit = generate_overparameterized_vqe_tfim_circuit(
        number_of_qubits, number_of_layers, parameters
    )
elif CIRCUIT_TYPE == "j1j2":
    number_of_parameters = (
        (3 * (2 * number_of_qubits - 3)) + number_of_qubits
    ) * number_of_layers
    parameters = [
        sympy.Symbol("theta{}".format(i)) for i in range(number_of_parameters)
    ]
    parameterized_quantum_circuit = generate_overparameterized_vqe_j1j2_circuit(
        number_of_qubits, number_of_layers, parameters
    )
elif CIRCUIT_TYPE == "j1j2_alternating-ansatz":
    number_of_parameters = ((2 * 3)) * number_of_layers
    parameters = [
        sympy.Symbol("theta{}".format(i)) for i in range(number_of_parameters)
    ]
    parameterized_quantum_circuit = generate_alternating_vqe_j1j2_circuit(
        number_of_qubits, number_of_layers, parameters
    )
assert number_of_parameters == len(parameterized_quantum_circuit.free_symbols)


energies = DATA[str(number_of_qubits)].get(str(number_of_layers), [])
print("------Running {} Qubits".format(number_of_qubits))
print("-----------Working on {} Layers".format(number_of_layers))
print(
    "---------------Need to run {} more trials".format(len(TRIAL_RANGE) - len(energies))
)


##### RUN TRIALS #####
for trial in TRIAL_RANGE:
    if trial >= len(energies):
        seed = (number_of_qubits * 123) + (number_of_layers * 97) + trial
        vqe_cost_function = get_vqe_cost_function(
            hamiltonian,
            parameterized_quantum_circuit,
            seed=seed,
            use_wandb=False,
        )

        initial_parameters = np.random.uniform(
            -1 * PARAMETER_PERIOD, PARAMETER_PERIOD, number_of_parameters
        )

        if OPTIMIZER == "L-BFGS-B":
            results = optimize_cost_function_with_lbfgsb(
                initial_parameters,
                vqe_cost_function,
                use_wandb=False,
                optimizer_options={"ftol": 1e-10},
            )
        elif OPTIMIZER == "CMA-ES":
            results = optimize_cost_function_with_cmaes(
                initial_parameters,
                vqe_cost_function,
                extra_config={},
                use_wandb=False,
                optimizer_options={
                    "sigma_0": 0.01,
                    "bounds": None,
                    "tolx": 1e-6,
                    "popsize": 36,
                    "maxfevals": 20000,
                },
            )

        energies.append(results.opt_value)

        DATA[str(number_of_qubits)][str(number_of_layers)] = energies

        if RECORD_DATA:
            with open(datafilename, "w") as f:
                f.write(json.dumps(DATA))
            f.close()
