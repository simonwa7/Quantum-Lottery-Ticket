from vqe.cost_function import get_vqe_cost_function
from vqe.circuits import (
    generate_alternating_vqe_j1j2_circuit,
)
from vqe.hamiltonians import generate_j1j2_hamiltonian
import sympy
import json
import numpy as np

datafilename = "data/v2/weight_decay/VQE-J1J2_J2=1.25_alternating-ansatz.json"
with open(datafilename, "r") as f:
    DATA = json.loads(f.read())

BOUNDARY_CONDITIONS = "open"
CIRCUIT_TYPE = "alternating-ansatz"
J2 = 1.25
NUMBER_OF_QUBITS = 5
NUMBER_OF_LAYERS = 8


hamiltonian = generate_j1j2_hamiltonian(NUMBER_OF_QUBITS, J2, j1=1)

number_of_parameters = (
    (3 * (NUMBER_OF_QUBITS - 1)) + NUMBER_OF_QUBITS
) * NUMBER_OF_LAYERS
parameters = [sympy.Symbol("theta{}".format(i)) for i in range(number_of_parameters)]
parameterized_quantum_circuit = generate_alternating_vqe_j1j2_circuit(
    NUMBER_OF_QUBITS, NUMBER_OF_LAYERS, parameters
)
assert number_of_parameters == len(parameterized_quantum_circuit.free_symbols)

for weight_decay in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]:
    print("---- {} ----".format(weight_decay))
    weight_decay_data = DATA[str(NUMBER_OF_QUBITS)][str(NUMBER_OF_LAYERS)].get(
        str(weight_decay),
        {"costs": [], "optimal_parameter_vectors": [], "energies": []},
    )
    weight_decay_data["energies"] = []

    unweighted_cost_function = get_vqe_cost_function(
        hamiltonian,
        parameterized_quantum_circuit,
        use_wandb=False,
        weight_decay=0,
    )
    weighted_cost_function = get_vqe_cost_function(
        hamiltonian,
        parameterized_quantum_circuit,
        use_wandb=False,
        weight_decay=weight_decay,
    )

    for cost, parameter_vector in zip(
        weight_decay_data["costs"], weight_decay_data["optimal_parameter_vectors"]
    ):
        parameter_vector = np.asarray(parameter_vector)
        assert cost == weighted_cost_function(parameter_vector)
        energy = unweighted_cost_function(parameter_vector)
        weight_decay_data["energies"].append(energy)

    DATA[str(NUMBER_OF_QUBITS)][str(NUMBER_OF_LAYERS)][
        str(weight_decay)
    ] = weight_decay_data

    with open(datafilename, "w") as f:
        f.write(json.dumps(DATA))
    f.close()
