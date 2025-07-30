"""
Classes to represent simulations.
"""
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator


class Simulation(ABC):
    """
    Abstract representation of a simulation.
    """

    @abstractmethod
    def __init__(self):
        pass

    @property
    @abstractmethod
    def n_params(self) -> int:
        """
        Number of model parameters.
        """

    @property
    @abstractmethod
    def n_data(self) -> int:
        """
        Number of data values.
        """

    @abstractmethod
    def __call__(self, model) -> NDArray[np.float64]:
        """
        Evaluate simulation for a given model.
        """

    @abstractmethod
    def jacobian(self, model) -> NDArray[np.float64] | LinearOperator:
        """
        Jacobian matrix for a given model.
        """
