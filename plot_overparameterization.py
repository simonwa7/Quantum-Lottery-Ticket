import matplotlib.pyplot as plt
import json

ENERGY_DIFFERENCE_CUTOFF = 1e-7

plt.rc("font", size=10)  # controls default text size
plt.rc("axes", titlesize=10)  # fontsize of the title
plt.rc("axes", labelsize=10)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=10)  # fontsize of the x tick labels
plt.rc("ytick", labelsize=10)  # fontsize of the y tick labels
plt.rc("legend", fontsize=10)  # fontsize of the legend

CIRCUIT_TYPE = "j1j2_alternating-ansatz-J2=1.25"
with open("data/overparameterization_{}.json".format(CIRCUIT_TYPE), "r") as f:
    data = json.loads(f.read())

qubit_counts = [int(key) for key in data.keys()]
ncols = 1
nrows = 1
fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(8, 8),
)

ax = axes
ax.set_xlabel("Number of Ansatz Layers")
ax.set_ylabel("Success Probability")
ax.set_title(
    "Presence of Phase Transition in Optimization of {} PQC for VQE".format(
        CIRCUIT_TYPE
    )
)

for number_of_qubits in qubit_counts:
    ground_state_energy = data[str(number_of_qubits)]["ground_state_energy"]
    numbers_of_layers = list(data[str(number_of_qubits)].keys())
    numbers_of_layers.remove("ground_state_energy")

    success_probabilities = []
    numbers_of_layers_to_plot = []
    for number_of_layers in numbers_of_layers:
        energies = data[str(number_of_qubits)][number_of_layers]
        number_of_layers = int(number_of_layers)

        count_within_cutoff = 0
        for energy in energies:
            if (energy - ground_state_energy) < ENERGY_DIFFERENCE_CUTOFF:
                count_within_cutoff += 1

        if len(energies) > 0:
            success_probabilities.append(count_within_cutoff / len(energies))
            numbers_of_layers_to_plot.append(number_of_layers)
    ax.scatter(
        numbers_of_layers_to_plot,
        success_probabilities,
        lw=2,
        label=str(number_of_qubits) + " Qubit {}".format(CIRCUIT_TYPE),
    )
    ax.plot(numbers_of_layers_to_plot, success_probabilities, lw=0.5, ls="dashed")

ax.legend()


# plt.show()
plt.savefig(
    "figures/overparameterization_{}.pdf".format(CIRCUIT_TYPE),
    dpi=300,
)
