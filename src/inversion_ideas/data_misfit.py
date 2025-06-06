"""
Class to represent a data misfit term.
"""
import numpy as np
from .objective_function import Objective


class DataMisfit(Objective):
    """
    L2 data misfit term.
    """

    def __init__(self, data, weights, simulation):
        self.data = data
        self.weights = weights
        self.simulation = simulation

    def __call__(self, model) -> float:  # noqa: D102
        residual = self.residual(model)
        return np.sum(residual.T @ residual)

    def gradient(self):
        """
        Number of model parameters.
        """
        return self.simulation.n_params

    @property
    def n_params(self):
        """
        Number of model parameters.
        """
        return self.simulation.n_params

    def residual(self, model):
        """
        Residual.
        """
        return (self.data - self.simulation(model)) / self.weights
