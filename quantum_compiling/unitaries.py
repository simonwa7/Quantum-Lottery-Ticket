import cirq
import numpy as np


def get_target_unitary(parameters):
    qubits = [cirq.LineQubit(i) for i in range(2)]
    circuit = cirq.Circuit()

    circuit.append(cirq.ry(parameters[0]).on(qubits[0]))
    circuit.append(cirq.CNOT(qubits[0], qubits[1]))
    circuit.append(cirq.ry(parameters[1]).on(qubits[1]))
    return circuit.unitary()


def get_overparameterized_unitary(parameters, number_of_layers):
    qubits = [cirq.LineQubit(i) for i in range(2)]
    circuit = cirq.Circuit()

    for i in range(number_of_layers):
        circuit.append(cirq.ry(parameters[(11 * i) + 0]).on(qubits[0]))
        circuit.append(cirq.ry(parameters[(11 * i) + 1]).on(qubits[1]))
        circuit.append(cirq.CNOT(qubits[0], qubits[1]))
        circuit.append(cirq.ry(parameters[(11 * i) + 2]).on(qubits[0]))
        circuit.append(cirq.ry(parameters[(11 * i) + 3]).on(qubits[1]))
        circuit.append(cirq.rz(parameters[(11 * i) + 4]).on(qubits[0]))
        circuit.append(cirq.rz(parameters[(11 * i) + 5]).on(qubits[1]))
        # Normalizing every parameter to the period of 4pi
        circuit.append(
            cirq.XXPowGate(exponent=parameters[(11 * i) + 6] / (2 * np.pi)).on(
                qubits[0], qubits[1]
            )
        )
        circuit.append(cirq.ry(parameters[(11 * i) + 7]).on(qubits[0]))
        circuit.append(cirq.ry(parameters[(11 * i) + 8]).on(qubits[1]))
        circuit.append(cirq.rz(parameters[(11 * i) + 9]).on(qubits[0]))
        circuit.append(cirq.rz(parameters[(11 * i) + 10]).on(qubits[1]))

    return circuit.unitary()
