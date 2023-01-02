import numpy as np


def get_parameter_indices_to_be_pruned(parameters, cutoff, period):
    return [
        index
        for index, parameter in enumerate(parameters)
        if np.isclose(0.0, parameter, atol=cutoff)
        or np.isclose(period, parameter, atol=cutoff)
    ]


def get_parameter_indices_to_be_pruned_using_percentage(
    parameters, percentage_to_prune, period
):
    number_of_parameters_to_prune = np.ceil(len(parameters) * percentage_to_prune)

    magnitudes_from_nearest_zero = []
    for parameter in parameters:
        remainder = parameter % period
        magnitudes_from_nearest_zero.append(
            min(
                abs(0.0 - remainder),
                abs(period - remainder),
            )
        )

    pruned_indices = []
    while len(pruned_indices) < number_of_parameters_to_prune:
        min_magnitude = np.inf
        min_index = None
        for index, magnitude in enumerate(magnitudes_from_nearest_zero):
            if index in pruned_indices:
                continue

            if magnitude < min_magnitude:
                min_magnitude = magnitude
                min_index = index
        pruned_indices.append(min_index)

    return sorted(pruned_indices)


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
