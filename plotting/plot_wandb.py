import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use("classic")
import numpy as np
import pandas as pd

number_of_layers = 16
sns.set()

dataframe = pd.read_csv(
    "/Users/williamsimon/Desktop/Research/Quantum-Lottery-Ticket/data/qlt/J1J2-VQE/QLT-VQE-J1J2-v0.5.csv"
)
dataframe = dataframe[(dataframe["number_of_layers"] == number_of_layers)]

plot = sns.lineplot(
    data=dataframe, x="Step", y="Offset Energy", hue="pruning", errorbar="se"
)
plot.set_yscale("log")
# plot.set_yticks([1e-2, 1, 10])
# plot.set_yticklabels([1e-2, 1, 10])
plt.show()
