"""Before saying anything, we want to note that success probability and parameter % are uncorrelated
in these plots in the sense that the accuracy threshold constant determines the shape of the success 
probability curve, while the parameter mag threshold constant determines the shape of the parameter 
curve. 

What we want to see in these plots is a weight decay value in which we can find a jump in parameter 
% being below the threshold (jump in comparison to lower WD's) where the success probability is still
relatively high for cutoffs around 1e-4 to 1e-6. The rationale for looking at the "jump" is that we 
want the WD parameter to actually contribute to driving more parameters to zero, not just getting
more parameters below the threshold as the threshold is raised and lowered (by chance). The rationale
for the criteria in accuracy is that we want to get a WD that can drive parameters to zero without 
infringing on the accuracy of the opitmized energy.

A next step would be to plot success probability of pruned circuits: take optimal parameter vectors, 
prune to threshold, calculate associated energy, and then plot success % of these energies to make sure
that pruning threshold doesn't impact energy too much after pruning. """


import matplotlib.pyplot as plt
import json

plt.rc("font", size=10)  # controls default text size
plt.rc("axes", titlesize=10)  # fontsize of the title
plt.rc("axes", labelsize=10)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=10)  # fontsize of the x tick labels
plt.rc("ytick", labelsize=10)  # fontsize of the y tick labels
plt.rc("legend", fontsize=8)  # fontsize of the legend

NUMBER_OF_QUBITS = 5
NUMBER_OF_LAYERS = 8
ACCURACY_THRESHOLD = 1e-4
PERIOD = 2 * 3.14159
# PARAMETER_MAGNITUDE_THRESHOLD_PERCENTAGE = 1

with open(
    "data/v2/weight_decay/VQE-J1J2_J2=1.25_alternating-ansatz.json",
    "r",
) as f:
    data = json.loads(f.read())


length = 6 / 2.54  # convert inches to cm
width = 8 / 2.54  # convert inches to cm
fig = plt.figure(figsize=(width, length), tight_layout=True)
success_probability_ax = plt.subplot(111)
# success_probability_ax.set_title(
#     "Parameter Magnitude Cutoff: {}%".format(PARAMETER_MAGNITUDE_THRESHOLD_PERCENTAGE)
# )

success_probability_color = "#E69F00"
success_probability_ax.set_xlabel("Weight Decay ($\Delta$)")
success_probability_ax.set_ylabel(
    "Success Probability", color=success_probability_color
)
success_probability_ax.set_ylim(-0.05, 1.05)

parameter_color = "#4668ea"
parameter_ax = success_probability_ax.twinx()
parameter_ax.set_ylabel("% Parameters", color=parameter_color)
parameter_ax.set_ylim(-5, 105)


for t in success_probability_ax.get_yticklabels():
    t.set_color(success_probability_color)

for t in parameter_ax.get_yticklabels():
    t.set_color(parameter_color)


ground_state_energy = data[str(NUMBER_OF_QUBITS)]["ground_state_energy"]

# colors = ["#009E73", "#0072B2", "#D55E00", "#CC79A7"]
for parameter_magnitude_threshold_percentage, style in zip(
    [10, 5, 2.5, 1], ["solid", "dotted", "dashed", "dashdot"]
):

    PARAMETER_MAGNITUDE_THRESHOLD = (
        parameter_magnitude_threshold_percentage / 100
    ) * PERIOD
    weight_decays = []
    success_probabilities = []
    parameter_percentages = []
    for weight_decay in data[str(NUMBER_OF_QUBITS)][str(NUMBER_OF_LAYERS)]:
        energies = data[str(NUMBER_OF_QUBITS)][str(NUMBER_OF_LAYERS)][weight_decay][
            "energies"
        ]
        if len(energies) == 0:
            continue
        optimal_parameter_vectors = data[str(NUMBER_OF_QUBITS)][str(NUMBER_OF_LAYERS)][
            weight_decay
        ]["optimal_parameter_vectors"]

        count_within_cutoff = 0
        for energy in energies:
            if abs(energy - ground_state_energy) < ACCURACY_THRESHOLD:
                count_within_cutoff += 1
        success_probabilities.append(count_within_cutoff / len(energies))
        trial_percentages = []
        for parameters in optimal_parameter_vectors:
            parameter_count_below_cutoff = 0
            for parameter in parameters:
                remainder = parameter % PERIOD
                magnitude = min(
                    abs(0.0 - remainder),
                    abs(PERIOD - remainder),
                )
                if magnitude < PARAMETER_MAGNITUDE_THRESHOLD:
                    parameter_count_below_cutoff += 1
            trial_percentages.append(
                100 * (parameter_count_below_cutoff / len(parameters))
            )
        parameter_percentages.append(sum(trial_percentages) / len(trial_percentages))
        weight_decays.append(float(weight_decay))

    parameter_ax.plot(
        weight_decays,
        parameter_percentages,
        color=parameter_color,
        ls=style,
        label="{}%".format(parameter_magnitude_threshold_percentage),
    )
    parameter_ax.scatter(
        weight_decays,
        parameter_percentages,
        color=parameter_color,
        s=5,
    )

success_probability_ax.scatter(
    weight_decays,
    success_probabilities,
    s=5,
    label="Success Probability",
    color=success_probability_color,
)
success_probability_ax.plot(
    weight_decays,
    success_probabilities,
    label="Success Probability",
    color=success_probability_color,
)
plt.xscale("log")
parameter_ax.legend()

plt.tight_layout()
plt.savefig(
    "figures/v2/weight-decay-hyperparameter-tuning_J1J2_J2=1.25_{}-qubits_{}-layers_Accuracy-Threshold={}.pdf".format(
        NUMBER_OF_QUBITS,
        NUMBER_OF_LAYERS,
        ACCURACY_THRESHOLD,
    ),
    dpi=300,
)
