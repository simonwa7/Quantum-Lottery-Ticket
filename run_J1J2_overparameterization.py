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
OPTIMIZER = "L-BFGS-B"

datafilename = "data/v2/overparameterization/VQE-J1J2_J2={}_{}.json".format(
    J2, CIRCUIT_TYPE
)
try:
    with open(datafilename, "r") as f:
        DATA = json.loads(f.read())
    f.close()
except:
    DATA = {}
    with open(datafilename, "w") as f:
        f.write(json.dumps(DATA))
    f.close()

number_of_qubits = int(sys.argv[1])

##### GENERATE HAMILTONIAN #####
qubit_data = DATA.get(str(number_of_qubits), {})
DATA[str(number_of_qubits)] = qubit_data

hamiltonian = generate_j1j2_hamiltonian(number_of_qubits, J2, j1=1)

ground_state_energy = eigenspectrum(hamiltonian)[0]
DATA[str(number_of_qubits)]["ground_state_energy"] = ground_state_energy


##### GENERATE PARAMETERIZED QUANTUM CIRCUIT #####
for number_of_layers in [3]:
    print("---- {} Layers ----".format(number_of_layers))
    number_of_parameters = (
        (3 * (number_of_qubits - 1)) + number_of_qubits
    ) * number_of_layers
    parameters = [
        sympy.Symbol("theta{}".format(i)) for i in range(number_of_parameters)
    ]
    parameterized_quantum_circuit = generate_alternating_vqe_j1j2_circuit(
        number_of_qubits, number_of_layers, parameters
    )
    assert number_of_parameters == len(parameterized_quantum_circuit.free_symbols)

    layer_data = DATA[str(number_of_qubits)].get(
        str(number_of_layers), {"energies": [], "optimal_parameter_vectors": []}
    )
    print(
        "     Need to run {} more trials".format(
            len(TRIAL_RANGE) - len(layer_data["energies"])
        )
    )

    ##### RUN TRIALS #####
    for trial in TRIAL_RANGE:
        if trial >= len(layer_data["energies"]):
            seed = (number_of_qubits * 123) + (number_of_layers * 97) + trial
            vqe_cost_function = get_vqe_cost_function(
                hamiltonian,
                parameterized_quantum_circuit,
                seed=seed,
                use_wandb=False,
            )

            initial_parameters = np.random.uniform(
                -1 * (PARAMETER_PERIOD / 2),
                (PARAMETER_PERIOD / 2),
                number_of_parameters,
            )

            results = optimize_cost_function_with_lbfgsb(
                initial_parameters,
                vqe_cost_function,
                use_wandb=False,
                optimizer_options={"ftol": 1e-10},
            )

            layer_data["energies"].append(results.opt_value)
            layer_data["optimal_parameter_vectors"].append(results.opt_params.tolist())

            DATA[str(number_of_qubits)][str(number_of_layers)] = layer_data

            if RECORD_DATA:
                with open(datafilename, "w") as f:
                    f.write(json.dumps(DATA))
                f.close()
