"""
Classes to represent simulations.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator

from ..typing import Model


class Simulation(ABC):
    """
    Abstract representation of a simulation.
    """

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
    def __call__(self, model: Model) -> NDArray[np.float64]:
        """
        Evaluate simulation for a given model.

        Parameters
        ----------
        model : (n_params) array
            Array with model values.

        Returns
        -------
        (n_data)
            Array with predicted data values for the given model.
        """

    @abstractmethod
    def jacobian(self, model: Model) -> NDArray[np.float64] | LinearOperator:
        """
        Jacobian matrix for a given model.

        Parameters
        ----------
        model : (n_params) array
            Array with model values.

        Returns
        -------
        (n_data, n_params) array or LinearOperator
            Jacobian matrix as a dense or sparse array,
            or as a :class:`~scipy.sparse.linalg.LinearOperator`.
        """
