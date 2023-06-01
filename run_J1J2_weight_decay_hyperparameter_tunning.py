from vqe.cost_function import get_vqe_cost_function
from vqe.circuits import generate_alternating_vqe_j1j2_circuit
from vqe.hamiltonians import generate_j1j2_hamiltonian
from optimize import optimize_cost_function_with_lbfgsb

from openfermion.linalg import eigenspectrum
import sympy
import numpy as np
import json

RECORD_DATA = True
TRIAL_RANGE = range(10)
BOUNDARY_CONDITIONS = "open"
CIRCUIT_TYPE = "alternating-ansatz"
PARAMETER_PERIOD = 2 * np.pi
J2 = 1.25
OPTIMIZER = "L-BFGS-B"
NUMBER_OF_QUBITS = 5
NUMBER_OF_LAYERS = 8

print("{} Qubits {} Layers".format(NUMBER_OF_QUBITS, NUMBER_OF_LAYERS))

datafilename = "data/v2/weight_decay/VQE-J1J2_J2={}_{}.json".format(J2, CIRCUIT_TYPE)
try:
    with open(datafilename, "r") as f:
        DATA = json.loads(f.read())
    f.close()
except:
    DATA = {}
    with open(datafilename, "w") as f:
        f.write(json.dumps(DATA))
    f.close()

hamiltonian = generate_j1j2_hamiltonian(NUMBER_OF_QUBITS, J2, j1=1)
ground_state_energy = eigenspectrum(hamiltonian)[0]
QUBIT_DATA = DATA.get(str(NUMBER_OF_QUBITS), {})
DATA[str(NUMBER_OF_QUBITS)] = QUBIT_DATA
QUBIT_DATA["ground_state_energy"] = ground_state_energy


number_of_parameters = (
    (3 * (NUMBER_OF_QUBITS - 1)) + NUMBER_OF_QUBITS
) * NUMBER_OF_LAYERS
parameters = [sympy.Symbol("theta{}".format(i)) for i in range(number_of_parameters)]
parameterized_quantum_circuit = generate_alternating_vqe_j1j2_circuit(
    NUMBER_OF_QUBITS, NUMBER_OF_LAYERS, parameters
)
assert number_of_parameters == len(parameterized_quantum_circuit.free_symbols)

LAYER_DATA = QUBIT_DATA.get(str(NUMBER_OF_LAYERS), {})
DATA[str(NUMBER_OF_QUBITS)][str(NUMBER_OF_LAYERS)] = LAYER_DATA
for weight_decay in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]:
    print("---- {} ----".format(weight_decay))
    weight_decay_data = LAYER_DATA.get(
        str(weight_decay),
        {"costs": [], "optimal_parameter_vectors": [], "number_of_evaluations": []},
    )
    print(
        "      Need to run {} more trials".format(
            len(TRIAL_RANGE) - len(weight_decay_data["costs"])
        )
    )

    ##### RUN TRIALS #####
    for trial in TRIAL_RANGE:
        if trial >= len(weight_decay_data["costs"]):
            seed = (
                (NUMBER_OF_QUBITS * 123)
                + (NUMBER_OF_LAYERS * 97)
                + trial
                + (weight_decay * 97541355 / 785)
            )
            vqe_cost_function = get_vqe_cost_function(
                hamiltonian,
                parameterized_quantum_circuit,
                seed=seed,
                use_wandb=False,
                weight_decay=weight_decay,
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

            weight_decay_data["costs"].append(results.opt_value)
            weight_decay_data["optimal_parameter_vectors"].append(
                results.opt_params.tolist()
            )
            weight_decay_data["number_of_evaluations"].append(results.nfev)

            DATA[str(NUMBER_OF_QUBITS)][str(NUMBER_OF_LAYERS)][
                str(weight_decay)
            ] = weight_decay_data

            if RECORD_DATA:
                with open(datafilename, "w") as f:
                    f.write(json.dumps(DATA))
                f.close()
            print("    Finished Trial ", trial)
