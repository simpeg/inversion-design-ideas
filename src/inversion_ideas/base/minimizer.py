"""
Base class for minimizer.
"""

from abc import ABC, abstractmethod
from collections.abc import Generator

from ..typing import Model
from .objective_function import Objective


class Minimizer(ABC):
    """
    Base class to represent minimizers as generators.
    """

    @abstractmethod
    def __call__(self, objective: Objective, initial_model: Model) -> Generator[Model]:
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
        Generator[Model]
            Generator that yields models after each iteration of the minimizer.
        """
