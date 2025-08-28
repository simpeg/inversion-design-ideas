"""
Classes to define minimizers.
"""

from typing import Callable
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import sparray
from scipy.sparse.linalg import LinearOperator, cg

from .base import Minimizer, Objective
from .errors import ConvergenceWarning


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
        self,
        objective: Objective,
        initial_model: NDArray[np.float64],
        preconditioner: NDArray[np.float64] | sparray | LinearOperator | None = None,
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
        preconditioner : (n_params, n_params) array, sparray or LinearOperator or Callable, optional
            Matrix used as preconditioner in the conjugant gradient algorithm.
            If None, no preconditioner will be used.
            A callable can be passed to build the preconditioner dynamically: such
            callable should take a single ``initial_model`` argument and return an
            array, `sparray` or a `LinearOperator`.

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
        if preconditioner is not None and "M" in self.cg_kwargs:
            msg = (
                "Cannot simultanously set `preconditioner` "
                "if `M` was passed in `cg_kwargs."
            )
            raise ValueError(msg)

        kwargs = self.cg_kwargs.copy()
        if preconditioner is not None:
            if isinstance(preconditioner, Callable):
                preconditioner = preconditioner(initial_model)
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
