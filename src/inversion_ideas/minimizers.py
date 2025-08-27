"""
Classes to define minimizers.
"""

from typing import Callable
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import sparray
from scipy.sparse.linalg import cg

from .base import Minimizer, Objective
from .errors import ConvergenceWarning


class ConjugateGradient(Minimizer):
    """
    Conjugate gradient minimizer.

    Parameters
    ----------
    preconditioner_callback : Callable or None, optional
        Callable used to create a preconditioner before running the minimization.
        The callable should take a single ``model`` argument.
        When the minimizer gets called, the new preconditioner is built using this
        callback, and used in the conjugate gradient to run the minimization.
    cg_kwargs :
        Additional arguments to be passed to :func:`scipy.sparse.linalg.cg`.
    """

    def __init__(self, preconditioner_callback: Callable | None = None, **cg_kwargs):
        if preconditioner_callback is not None and "M" in cg_kwargs:
            msg = (
                "Cannot simultanously pass `preconditioner_callback` and `M`. "
                "Choose either a static preconditioner by passing an `M` argument, or "
                "a dynamic one through `preconditioner_callback`."
            )
            raise ValueError(msg)

        self.preconditioner_callback = preconditioner_callback
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
        kwargs = self.cg_kwargs.copy()

        # Build preconditioner (if any)
        preconditioner = self._get_preconditioner(initial_model)
        if preconditioner is not None:
            kwargs["M"] = preconditioner

        # TODO: maybe it would be nice to add a `is_linear` attribute to the objective
        # functions for the ones that generate a linear problem.
        gradient = objective.gradient(initial_model)
        hessian = objective.hessian(initial_model)
        model_step, info = cg(hessian, -gradient, **kwargs)
        if info != 0:
            warnings.warn(
                "Conjugate gradient convergence to tolerance not achieved after "
                f"{info} number of iterations.",
                ConvergenceWarning,
                stacklevel=2,
            )
        inverted_model = initial_model + model_step
        return inverted_model

    def _get_preconditioner(self, model) -> sparray | None:
        """
        Build the preconditioner.
        """
        if self.preconditioner_callback is not None and "M" in self.cg_kwargs:
            msg = (
                "Cannot simultanously set `preconditioner_callback` and `M`. "
                "Choose either a static preconditioner by passing an `M` argument, or "
                "a dynamic one through `preconditioner_callback`."
            )
            raise ValueError(msg)

        if self.preconditioner_callback is None:
            return None

        return self.preconditioner_callback(model)
