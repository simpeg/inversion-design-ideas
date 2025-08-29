"""
Base class for minimizer.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from .objective_function import Objective


class Minimizer(ABC):
    """
    Base class to represent minimizers.
    """

    @abstractmethod
    def __call__(
        self, objective: Objective, initial_model: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Minimize objective function.

        Parameters
        ----------
        objective : Objective
            Objective function to be minimized.
        initial_model : (n_params) array
            Initial model used to start the minimization.

        Returns
        -------
        inverted_model : (n_params) array
           Inverted model obtained after minimization.
        """
