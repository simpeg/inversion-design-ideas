"""
Classes to define minimizers.
"""

import warnings
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import line_search
from scipy.sparse import sparray
from scipy.sparse.linalg import LinearOperator, cg

from .base import Minimizer, Objective
from .errors import ConvergenceWarning


class ConjugateGradient(Minimizer):
    """
    Conjugate gradient minimizer.
    """

    def __call__(
        self,
        objective: Objective,
        initial_model: NDArray[np.float64],
        preconditioner: NDArray[np.float64] | sparray | LinearOperator | None = None,
        **kwargs,
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
        """  # noqa: E501
        if preconditioner is not None and "M" in kwargs:
            msg = "Cannot simultanously pass `preconditioner` and `M`."
            raise ValueError(msg)

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


class GaussNewtonConjugateGradient(Minimizer):
    """
    Gauss-Newton conjugate gradient minimizer.
    """

    def __call__(
        self,
        objective: Objective,
        initial_model: NDArray[np.float64],
        preconditioner: NDArray[np.float64] | sparray | LinearOperator | None = None,
        maxiter_gauss_newton: int = 100,
        maxiter_line_search: int = 10,
        **cg_kwargs,
    ) -> NDArray[np.float64]:
        r"""
        Minimize objective function with a Gauss-Newton Conjugate Gradient method.

        Performs a line search using the Wolfe conditions to obtain the next step in the
        model space.

        .. important::

            This minimizer should be used only for non-linear objective functions.

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
        cg_kwargs : dict
            Extra arguments that will be passed to the
            :func:`inversion_ideas.conjugate_gradient` function.

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
        """  # noqa: E501
        if preconditioner is not None and "M" in cg_kwargs:
            msg = "Cannot simultanously pass `preconditioner` and `M`."
            raise ValueError(msg)

        if preconditioner is not None:
            # Define a static preconditioner for all Gauss-Newton iterations
            if isinstance(preconditioner, Callable):
                preconditioner = preconditioner(initial_model)
            cg_kwargs["M"] = preconditioner

        # ----------------------------------------

        model = initial_model.copy()
        i = 0
        while True:
            if i >= maxiter_gauss_newton:
                msg = "Reached maximum number of iterations."
                raise RuntimeError(msg)

            # Apply CG
            gradient, hessian = objective.gradient(model), objective.hessian(model)
            model_step, info = cg(hessian, -gradient, **cg_kwargs)
            if info != 0:
                warnings.warn(
                    "Conjugate gradient convergence to tolerance not achieved after "
                    f"{info} number of iterations.",
                    ConvergenceWarning,
                    stacklevel=2,
                )

            print("Finished CG")

            # Perform line search using Wolfe condition
            line_search_result = line_search(
                objective,
                objective.gradient,
                model,
                model_step,
                maxiter=maxiter_line_search,
            )
            alpha = line_search_result[0]

            if alpha is None:
                msg = (
                    "Couldn't find a valid alpha, obtained None.\n"
                    + f"{line_search_result}"
                )
                raise RuntimeError(msg)

            print("Finished line search")

            # Update model
            model += alpha * model_step

            i += 1
        return model
