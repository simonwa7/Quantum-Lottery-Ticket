import numpy as np


def get_parameter_indices_to_be_pruned(parameters, cutoff, period):
    return [
        index
        for index, parameter in enumerate(parameters)
        if np.isclose(0.0, parameter, atol=cutoff)
        or np.isclose(period, parameter, atol=cutoff)
    ]


def get_pruned_parameters(unpruned_parameters, pruned_indices):
    pruned_parameters = unpruned_parameters
    for pruned_index in pruned_indices[::-1]:
        pruned_parameters = np.delete(pruned_parameters, pruned_index)
    return pruned_parameters


def get_padded_parameters(unpadded_parameters, pruned_indices):
    padded_parameters = unpadded_parameters
    for pruned_index in pruned_indices:
        padded_parameters = np.concatenate(
            (
                padded_parameters[:pruned_index],
                np.asarray([float(1e-15)]),
                padded_parameters[pruned_index:],
            )
        )
    return padded_parameters
