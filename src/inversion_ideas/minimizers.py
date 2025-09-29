"""
Classes to define minimizers.
"""

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
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
        maxiter: int = 100,
        maxiter_line_search: int = 10,
        rtol=1e-5,
        cg_kwargs: dict[str, Any] | None = None,
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
        maxiter : int, optional
            Maximum number of Gauss-Newton iterations.
        maxiter_line_search : int, optional
            Maximum number of line search iterations.
        rtol : float, optional
            Relative tolerance for the objective function. This value is used in the
            stopping criteria for the Gauss-Newton iterations.
        cg_kwargs : dict
            Extra arguments that will be passed to the
            :func:`inversion_ideas.conjugate_gradient` function.

        Returns
        -------
        inverted_model : (n_params) array
           Inverted model obtained after minimization.

        Notes
        -----
        TODO
        """  # noqa: E501
        cg_kwargs = cg_kwargs if cg_kwargs is not None else {}
        if preconditioner is not None and "M" in cg_kwargs:
            msg = "Cannot simultanously pass `preconditioner` and `M`."
            raise ValueError(msg)

        if preconditioner is not None:
            # Define a static preconditioner for all Gauss-Newton iterations
            if isinstance(preconditioner, Callable):
                preconditioner = preconditioner(initial_model)
            cg_kwargs["M"] = preconditioner

        # Perform Gauss-Newton iterations
        iteration = 0
        phi_prev_value = np.inf  # value of the objective function on previous model
        model = initial_model.copy()

        while True:
            # Stop if reached max number of iterations
            if iteration >= maxiter:
                msg = f"Reached maximum number of Gauss-Newton iterations ({maxiter})."
                raise RuntimeError(msg)

            # Check for stopping criteria
            phi_value = objective(model)
            if (
                not np.isinf(phi_prev_value)
                and np.abs(phi_value - phi_prev_value) <= phi_prev_value * rtol
            ):
                break

            # Apply Conjugate Gradient to get search direction
            gradient, hessian = objective.gradient(model), objective.hessian(model)
            search_direction, info = cg(hessian, -gradient, **cg_kwargs)
            if info != 0:
                warnings.warn(
                    "Conjugate gradient convergence to tolerance not achieved after "
                    f"{info} number of iterations.",
                    ConvergenceWarning,
                    stacklevel=2,
                )
            print("Finished CG")

            # Perform line search
            alpha, n_ls_iters = _backtracking_line_search(
                objective,
                model,
                search_direction,
                phi_value=phi_value,
                phi_gradient=gradient,
                maxiter=maxiter_line_search,
            )
            if alpha is None:
                msg = (
                    "Couldn't find a valid alpha, obtained None. "
                    f"Ran {n_ls_iters} iterations."
                )
                raise RuntimeError(msg)
            print(f"Finished line search in {n_ls_iters} iterations.")

            # Perform model step
            model += alpha * search_direction

            # Update cached values and iteration counter
            phi_prev_value = phi_value
            iteration += 1

        return model


def _backtracking_line_search(
    phi: Objective,
    model: NDArray[np.float64],
    search_direction: NDArray[np.float64],
    *,
    contraction_factor: float = 0.5,
    c_factor: float = 0.5,
    phi_value: float | None = None,
    phi_gradient: NDArray[np.float64] | None = None,
    maxiter: int = 20,
):
    """
    Implement the backtracking line search algorithm.

    Parameters
    ----------
    phi : Objective
        Objective function to which the line search will be applied.
    model : (n_params) array
        Current model.
    search_direction: (n_params) array
        Vector used as a search direction.
    contraction_factor : float
        Contraction factor for the step length. Must be greater than 0 and lower than 1.
    c_factor : float
        The c factor used in the descent condition.
        Must be greater than 0 and lower than 1.
    phi_value : float or None, optional
        Precomputed value of ``phi(model)``. If None, it will be computed.
    phi_gradient : (n_params) array, optional
        Precomputed value of ``phi.gradient(model)``. If None, it will be computed.
    maxiter : int, optional
        Maximum number of line search iterations.

    Returns
    -------
    step_length : float or None
        Alpha for which `x_new = x0 + alpha * pk`, or None if the line search algorithm
        did not converge.
    n_iterations : int
        Number of line search iterations.

    Notes
    -----
    TODO

    Nocedal & Wright (1999), page 41.

    References
    ----------
    Nocedal, J., & Wright, S. J. (1999). Numerical optimization. Springer.
    """
    phi_value = phi_value if phi_value is not None else phi(model)
    phi_gradient = phi_gradient if phi_gradient is not None else phi.gradient(model)

    def stop_condition(step_length):
        return (
            phi(model + step_length * search_direction)
            <= phi_value + c_factor * step_length * phi_gradient @ search_direction
        )

    step_length = 1.0
    n_iterations = 0
    while not stop_condition(step_length):
        step_length *= contraction_factor
        n_iterations += 1

        if n_iterations >= maxiter:
            return None, n_iterations

    return step_length, n_iterations
