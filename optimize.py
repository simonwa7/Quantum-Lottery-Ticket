from zquantum.optimizers import ScipyOptimizer, CMAESOptimizer
import wandb


def optimize_cost_function_with_lbfgsb(
    initial_parameters,
    cost_function,
    extra_config={},
    use_wandb=True,
    project="QLT-Deep-PoC",
    optimizer_options={"ftol": 1e-6},
):
    optimizer = ScipyOptimizer(method="L-BFGS-B", options=optimizer_options)

    if use_wandb:
        config = {
            **{
                "optimizer": "L-BFGS-B",
                "tolx": optimizer_options["ftol"],
                "initial_parameters": initial_parameters,
            },
            **extra_config,
        }

        run = wandb.init(project=project, config=config)

    results = optimizer.minimize(cost_function, initial_parameters, keep_history=True)

    if use_wandb:
        run.finish()

    return results


def optimize_cost_function_with_cmaes(
    initial_parameters,
    cost_function,
    extra_config={},
    use_wandb=True,
    project="QLT-Deep-PoC",
    optimizer_options={
        "sigma_0": 0.01,
        "bounds": None,
        "tolx": 1e-6,
        "popsize": 36,
    },
):
    sigma_0 = optimizer_options.pop("sigma_0")
    optimizer = CMAESOptimizer(
        sigma_0=sigma_0,
        options=optimizer_options,
    )

    if use_wandb:
        config = {
            **{
                "optimizer": "CMAES",
                "sigma_0": sigma_0,
                "bounds": optimizer_options.get("bounds", None),
                "tolx": optimizer_options.get("tolx", None),
                "popsize": optimizer_options.get("popsize", None),
                "initial_parameters": initial_parameters,
            },
            **extra_config,
        }

        run = wandb.init(project=project, config=config)

    results = optimizer.minimize(cost_function, initial_parameters, keep_history=True)

    if use_wandb:
        run.finish()

    return results
