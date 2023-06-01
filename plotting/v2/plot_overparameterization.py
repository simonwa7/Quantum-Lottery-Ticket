import matplotlib.pyplot as plt
import json

plt.rc("font", size=10)  # controls default text size
plt.rc("axes", titlesize=10)  # fontsize of the title
plt.rc("axes", labelsize=10)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=10)  # fontsize of the x tick labels
plt.rc("ytick", labelsize=10)  # fontsize of the y tick labels
plt.rc("legend", fontsize=8)  # fontsize of the legend

with open(
    "data/v2/overparameterization/VQE-J1J2_J2=1.25_j1j2_alternating-ansatz.json", "r"
) as f:
    data = json.loads(f.read())

qubit_counts = [int(key) for key in data.keys()]
ncols = 1
nrows = 1
fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(8 / 2.54, 6 / 2.54),
)

number_of_qubits = 4
ax = axes
ax.set_xlabel("Number of Ansatz Layers")
ax.set_ylabel("Success Probability")
ax.set_title("{} Qubit J1J2".format(number_of_qubits))
for i, energy_cutoff in enumerate([1e-4, 1e-6, 1e-8, 1e-10]):
    ground_state_energy = data[str(number_of_qubits)]["ground_state_energy"]
    numbers_of_layers = list(data[str(number_of_qubits)].keys())
    numbers_of_layers.remove("ground_state_energy")

    success_probabilities = []
    numbers_of_layers_to_plot = []
    for number_of_layers in numbers_of_layers:
        energies = data[str(number_of_qubits)][number_of_layers]["energies"]
        number_of_layers = int(number_of_layers)

        count_within_cutoff = 0
        for energy in energies:
            if (energy - ground_state_energy) < energy_cutoff:
                count_within_cutoff += 1

        if len(energies) > 0:
            success_probabilities.append(count_within_cutoff / len(energies))
            numbers_of_layers_to_plot.append(number_of_layers)
    marker_size = (75 - (24 * i)) * 10 / 10
    ax.scatter(
        numbers_of_layers_to_plot,
        success_probabilities,
        s=marker_size,
        label=energy_cutoff,
    )
    ax.plot(numbers_of_layers_to_plot, success_probabilities, lw=0.5, ls="dashed")
ax.legend()


plt.tight_layout()
plt.savefig(
    "figures/v2/overparameterization_J1J2_J2=1.25_{}-qubits.pdf".format(
        number_of_qubits
    ),
    dpi=300,
)
