import cirq
import numpy as np
from qecirq.conversions import import_from_cirq


def generate_alternating_vqe_j1j2_circuit(
    number_of_qubits, number_of_layers, parameters
):
    """Variational Hamiltonian Ansatz"""
    # assert 2 * number_of_layers
    qubits = [cirq.LineQubit(i) for i in range(number_of_qubits)]
    circuit = cirq.Circuit()

    for qubit_index in range(number_of_qubits):
        circuit.append(cirq.H(qubits[qubit_index]))

    parameter_counter = 0
    for _ in range(number_of_layers):
        # Add nearest-neighbor entanglers
        for qubit_index in range(0, number_of_qubits - 1):
            circuit = _add_xx_yy_zz_gate(
                circuit,
                qubits[qubit_index],
                qubits[qubit_index + 1],
                parameters[parameter_counter],
                parameters[parameter_counter + 1],
                parameters[parameter_counter + 2],
            )
        parameter_counter += 3

        # Add next nearest-neighbor entanglers
        for qubit_index in range(0, number_of_qubits - 2):
            circuit = _add_xx_yy_zz_gate(
                circuit,
                qubits[qubit_index],
                qubits[qubit_index + 2],
                parameters[parameter_counter],
                parameters[parameter_counter + 1],
                parameters[parameter_counter + 2],
            )
        parameter_counter += 3

        # # Add single qubit rotation layer
        # for qubit_index in range(0, number_of_qubits):
        #     circuit.append(
        #         cirq.rx(parameters[parameter_counter]).on(qubits[qubit_index])
        #     )
        #     parameter_counter += 1
    # from cirq.contrib.svg import circuit_to_svg

    # with open("circuit.svg", "w") as f:
    #     f.write(circuit_to_svg(circuit))
    # f.close()
    # import pdb

    # pdb.set_trace()
    return import_from_cirq(circuit)


def generate_overparameterized_vqe_j1j2_circuit(
    number_of_qubits, number_of_layers, parameters
):
    """Variational Hamiltonian Ansatz"""
    assert ((3 * (2 * number_of_qubits - 3)) + number_of_qubits) * number_of_layers
    qubits = [cirq.LineQubit(i) for i in range(number_of_qubits)]
    circuit = cirq.Circuit()

    for qubit_index in range(number_of_qubits):
        circuit.append(cirq.H(qubits[qubit_index]))

    parameter_counter = 0
    for _ in range(number_of_layers):
        # Add nearest-neighbor entanglers
        for qubit_index in range(0, number_of_qubits - 1):
            circuit = _add_xx_yy_zz_gate(
                circuit,
                qubits[qubit_index],
                qubits[qubit_index + 1],
                parameters[parameter_counter],
                parameters[parameter_counter + 1],
                parameters[parameter_counter + 2],
            )
            parameter_counter += 3

        # Add next nearest-neighbor entanglers
        for qubit_index in range(0, number_of_qubits - 2):
            circuit = _add_xx_yy_zz_gate(
                circuit,
                qubits[qubit_index],
                qubits[qubit_index + 2],
                parameters[parameter_counter],
                parameters[parameter_counter + 1],
                parameters[parameter_counter + 2],
            )
            parameter_counter += 3

        # Add single qubit rotation layer
        for qubit_index in range(0, number_of_qubits):
            circuit.append(
                cirq.rx(parameters[parameter_counter]).on(qubits[qubit_index])
            )
            parameter_counter += 1

    return import_from_cirq(circuit)


def generate_overparameterized_vqe_tfim_circuit(
    number_of_qubits, number_of_layers, parameters
):
    """Figure 7. Larocca, Martin, et al. "Theory of overparametrization in
    quantum neural networks." arXiv preprint arXiv:2109.11676 (2021)."""
    assert len(parameters) == 2 * (number_of_layers)
    qubits = [cirq.LineQubit(i) for i in range(number_of_qubits)]
    circuit = cirq.Circuit()

    for qubit_index in range(number_of_qubits):
        circuit.append(cirq.H(qubits[qubit_index]))

    for layer_index in range(number_of_layers):
        # Add even-pair entanglers
        for qubit_index in range(0, number_of_qubits, 2):
            if qubit_index >= number_of_qubits - 1:
                continue
            circuit = _add_zz_gate(
                circuit,
                qubits[qubit_index],
                qubits[qubit_index + 1],
                parameters[(2 * layer_index)],
            )

        # Add odd-pair entanglers
        for qubit_index in range(1, number_of_qubits - 1, 2):
            circuit = _add_zz_gate(
                circuit,
                qubits[qubit_index],
                qubits[qubit_index + 1],
                parameters[(2 * layer_index)],
            )

        # Add single qubit rotations
        for qubit_index in range(0, number_of_qubits):
            if parameters[(2 * layer_index) + 1] != 0:
                circuit.append(
                    cirq.rx(parameters[(2 * layer_index) + 1]).on(qubits[qubit_index])
                )

    return import_from_cirq(circuit)


def generate_2qubit_overparameterized_circuit(number_of_layers, parameters):
    """Eq. 30 Decomposition from Smith, Adam, et al. "Simulating quantum many-body
    dynamics on a current digital quantum computer." npj Quantum Information
    5.1 (2019): 1-13."""
    qubits = [cirq.LineQubit(i) for i in range(2)]
    circuit = cirq.Circuit()
    parameter_counter = 0
    for _ in range(number_of_layers):

        for qubit_index in range(0, 2):
            circuit.append(
                cirq.ry(parameters[parameter_counter]).on(qubits[qubit_index])
            )
            parameter_counter += 1
            circuit.append(
                cirq.rz(parameters[parameter_counter]).on(qubits[qubit_index])
            )
            parameter_counter += 1
            circuit.append(
                cirq.ry(parameters[parameter_counter]).on(qubits[qubit_index])
            )
            parameter_counter += 1

        circuit = _add_xx_yy_zz_gate(
            circuit,
            qubits[0],
            qubits[1],
            parameters[parameter_counter],
            parameters[parameter_counter + 1],
            parameters[parameter_counter + 2],
        )
        parameter_counter += 3

        for qubit_index in range(0, 2):
            circuit.append(
                cirq.ry(parameters[parameter_counter]).on(qubits[qubit_index])
            )
            parameter_counter += 1
            circuit.append(
                cirq.rz(parameters[parameter_counter]).on(qubits[qubit_index])
            )
            parameter_counter += 1
            circuit.append(
                cirq.ry(parameters[parameter_counter]).on(qubits[qubit_index])
            )
            parameter_counter += 1

    return import_from_cirq(circuit)


def _add_zz_gate(circuit, qubit1, qubit2, gamma):
    """Decomposition from Smith, Adam, et al. "Simulating quantum many-body
    dynamics on a current digital quantum computer." npj Quantum Information
    5.1 (2019): 1-13."""
    if gamma == 0:
        return circuit
    circuit.append(cirq.CNOT(qubit1, qubit2))
    circuit.append(cirq.rz(-2 * gamma).on(qubit2))
    circuit.append(cirq.CNOT(qubit1, qubit2))
    return circuit


def _add_xx_yy_zz_gate(circuit, qubit1, qubit2, alpha, beta, gamma):
    """Eq. 30 Decomposition from Smith, Adam, et al. "Simulating quantum many-body
    dynamics on a current digital quantum computer." npj Quantum Information
    5.1 (2019): 1-13.

    Corresponds to: exp(i (alpha x@x + beta y@y + gamma z@z)) (Eq. 28) - lambda renamed to omega

    period of parameters is 2*pi"""
    theta = (np.pi / 2) - gamma
    phi = alpha - (np.pi / 2)
    omega = (np.pi / 2) - beta

    circuit.append(cirq.rz(-np.pi / 2).on(qubit2))
    circuit.append(cirq.CNOT(qubit2, qubit1))
    circuit.append(cirq.rz(theta).on(qubit1))
    circuit.append(cirq.ry(phi).on(qubit2))
    circuit.append(cirq.CNOT(qubit1, qubit2))
    circuit.append(cirq.ry(omega).on(qubit2))
    circuit.append(cirq.CNOT(qubit2, qubit1))
    circuit.append(cirq.rz(np.pi / 2).on(qubit1))

    return circuit
