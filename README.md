# Quantum-Lottery-Ticket

## J1J2

For these experiments, we use the "alternating-ansatz" using vqe.circuits.`generate_alternating_vqe_j1j2_circuit` which is a variant of the Hamiltonian Variational Ansatz.

We also set J2=1.25. 

### Determining Overparameterization

#### By Number of Layers

In this method, follow the method proposed by Larocca et al to evaluate overparameterization in PQCs. For 4 and 5 qubits, we run 10 trials of VQE starting with random parameter initializations for different numbers of layers of the ansatz and determine the rate at which these trials achieve different accuracy levels.

The data generation for this is done using `run_J1J2_overparameterization.py`. Data is stored under `data/v2/overparameterization/`.

This data can be plotted using `plotting/v2/plot_overparameterization.py` and figures can be found in `figures/v2/`

<object data="./figures/v2/overparameterization_J1J2_J2=1.25_4-qubits.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="./figures/v2/overparameterization_J1J2_J2=1.25_4-qubits.pdf"></embed>
</object>

<object data="./figures/v2/overparameterization_J1J2_J2=1.25_5-qubits.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="./figures/v2/overparameterization_J1J2_J2=1.25_5-qubits.pdf"></embed>
</object>

#### By Pruning Percentage

TBD

### Weight Decay Hyperparameter Tuning

### Determining Pruning Threshold