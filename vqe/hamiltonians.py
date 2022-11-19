from openfermion import QubitOperator


def generate_tfim_hamiltonian(number_of_qubits, h=1, boundary_conditions="open"):
    """Page 6. Larocca, Martin, et al. "Theory of overparametrization in
    quantum neural networks." arXiv preprint arXiv:2109.11676 (2021)."""
    if boundary_conditions == "open":
        nf = number_of_qubits - 1
    elif boundary_conditions == "closed":
        nf = number_of_qubits

    hamiltonian = QubitOperator()

    for i in range(0, nf):
        hamiltonian -= QubitOperator("Z{} Z{}".format(i, i + 1))

    for i in range(0, number_of_qubits):
        hamiltonian -= QubitOperator("X{}".format(i), h)

    return hamiltonian


def generate_j1j2_hamiltonian(number_of_qubits, j2, j1=1):

    hamiltonian = QubitOperator()

    # Add nearest-neighbor terms
    for i in range(0, number_of_qubits - 1):
        hamiltonian += QubitOperator("X{} X{}".format(i, i + 1), j1)
        hamiltonian += QubitOperator("Y{} Y{}".format(i, i + 1), j1)
        hamiltonian += QubitOperator("Z{} Z{}".format(i, i + 1), j1)

    # Add next nearest-neighbor terms
    for i in range(0, number_of_qubits - 2):
        hamiltonian += QubitOperator("X{} X{}".format(i, i + 2), j2)
        hamiltonian += QubitOperator("Y{} Y{}".format(i, i + 2), j2)
        hamiltonian += QubitOperator("Z{} Z{}".format(i, i + 2), j2)

    return hamiltonian
