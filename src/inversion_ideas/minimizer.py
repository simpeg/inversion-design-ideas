"""
Classes to define minimizers.
"""
import warnings
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import cg

from .errors import ConvergenceWarning
from .objective_function import Objective


class Minimizer(ABC):
    """
    Base class to represent minimizers.
    """

    @abstractmethod
    def __init__(self):
        pass

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


class ConjugateGradient(Minimizer):
    """
    Conjugate gradient minimizer.

    Parameters
    ----------
    cg_kwargs :
        Additional arguments to be passed to :func:`scipy.sparse.linalg.cg`.
    """

    def __init__(self, **cg_kwargs):
        self.cg_kwargs = cg_kwargs

    def __call__(
        self, objective: Objective, initial_model: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""
        Minimize objective function with a Conjugate Gradient method.

        .. important::

            This minimizer should be used only for linear objective functions.

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

        Notes
        -----
        Minimize the objective function :math:`\phi(\mathbf{m})` by solving the system:

        .. math::

            \bar{\bar{\nabla}} \phi \mathbf{m}^{*} = - \bar{\nabla} \phi

        through a Conjugate Gradient algorithm, where :math:`\bar{\bar{\nabla}} \phi`
        and :math:`\bar{\nabla} \phi` are the the Hessian and the gradient of the
        objective function, respectively.
        """
        # TODO: maybe it would be nice to add a `is_linear` attribute to the objective
        # functions for the ones that generate a linear problem.
        gradient = objective.gradient(initial_model)
        hessian = objective.hessian(initial_model)
        inverted_model, info = cg(hessian, -gradient, **self.cg_kwargs)
        if info != 0:
            warnings.warn(
                "Conjugate gradient convergence to tolerance not achieved after "
                f"{info} number of iterations.",
                ConvergenceWarning,
                stacklevel=2,
            )
        return inverted_model
