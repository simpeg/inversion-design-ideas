"""
Base class for minimizer.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator

from ..typing import Model
from .objective_function import Objective


class MinimizerResult(dict):
    """
    Dictionary to store results of a single minimization iteration.

    This class is a child of ``dict``, but allows to access the values through
    attributes.

    Notes
    -----
    Inspired in the :class:`scipy.optimize.OptimizeResult`.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__  # type: ignore[assignment]
    __delattr__ = dict.__delitem__  # type: ignore[assignment]

    def __dir__(self):
        return list(self.keys())


class Minimizer(ABC):
    """
    Base class to represent minimizers as generators.
    """

    @abstractmethod
    def __call__(
        self,
        objective: Objective,
        initial_model: Model,
        *,
        callback: Callable[[MinimizerResult], None] | None = None,
    ) -> Generator[Model]:
        """
        Minimize objective function.

        Parameters
        ----------
        objective : Objective
            Objective function to be minimized.
        initial_model : (n_params) array
            Initial model used to start the minimization.
        callback : callable, optional
            Callable that gets called after each iteration.
            Takes a :class:`inversion_ideas.base.MinimizerResult` as argument.

        Returns
        -------
        Generator[Model]
            Generator that yields models after each iteration of the minimizer.
        """
