"""
Base class for minimizer.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from dataclasses import dataclass

from ..typing import Model
from .objective_function import Objective


@dataclass
class MinimizerResult:
    """Dataclass to store results of a single minimization iteration."""

    iteration: int
    model: Model
    conj_grad_iters: int | None = None
    line_search_iters: int | None = None
    step_norm: float | None = None


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
