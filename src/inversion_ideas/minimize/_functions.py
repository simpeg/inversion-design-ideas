"""
Minimizer functions.

Define functions that can be use to minimize an objective function in a single call.
"""

import warnings
from collections.abc import Callable

from scipy.sparse.linalg import cg

from ..base import Objective
from ..errors import ConvergenceWarning
from ..typing import Model, Preconditioner


def conjugate_gradient(
    objective: Objective,
    initial_model: Model,
    preconditioner: Preconditioner | Callable[[Model], Preconditioner] | None = None,
    **kwargs,
) -> Model:
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
    kwargs : dict
        Extra arguments that will be passed to the :func:`scipy.sparse.linalg.cg`
        function.

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
    if preconditioner is not None and "M" in kwargs:
        msg = "Cannot simultanously pass `preconditioner` and `M`."
        raise ValueError(msg)

    if preconditioner is not None:
        if callable(preconditioner):
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
