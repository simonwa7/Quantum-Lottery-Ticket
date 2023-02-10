import pandas as pd
import wandb
import sys
import os

entity, project = "simonwa7", sys.argv[1]
api = wandb.Api()
api = wandb.Api(timeout=100)

if "J1J2" in project:
    data_dir = (
        "/Users/williamsimon/Desktop/Research/Quantum-Lottery-Ticket/data/qlt/J1J2-VQE/"
    )
elif "QCBM" in project:
    data_dir = (
        "/Users/williamsimon/Desktop/Research/Quantum-Lottery-Ticket/data/qlt/QCBM/"
    )

runs = api.runs(entity + "/" + project)

if os.path.exists(data_dir + "{}.csv".format(project)):
    dataframe = pd.read_csv(data_dir + "{}.csv".format(project))
else:
    dataframe = pd.DataFrame({"Name": []})


for i, run in enumerate(runs):
    print("[{}%] ".format(round(100 * (i / len(runs)), 1)), run.name)
    if len(dataframe[dataframe["Name"] == run.name]) == 0:
        del run.config["initial_parameters"]
        del run.config["unpruned_initial_parameters"]
        for iteration in run.scan_history():
            current_run_iteration = {
                **dict(run.config),
                **dict(iteration),
                **{"Name": run.name, "Step": iteration["_step"]},
            }
            dataframe = dataframe.append(
                current_run_iteration, ignore_index=True
            )
        dataframe.to_csv(data_dir + "{}.csv".format(project))
