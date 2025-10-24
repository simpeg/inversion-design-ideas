"""
Minimizer classes.
"""

import warnings
from collections.abc import Callable, Generator
from typing import Any

import numpy as np
from scipy.sparse.linalg import cg

from ..base import Condition, Minimizer, Objective
from ..errors import ConvergenceWarning
from ..typing import Model, Preconditioner
from ..utils import get_logger
from ._utils import backtracking_line_search


class GaussNewtonConjugateGradient(Minimizer):
    """
    Minimize non-linear objective functions using a Gauss-Newton Conjugate Gradient.

    Apply Gauss-Newton iterations using a Conjugate Gradient to find search directions,
    and use a backtracking line search to update the model after each iteration.

    Parameters
    ----------
    maxiter : int, optional
        Maximum number of Gauss-Newton iterations.
    maxiter_line_search : int, optional
        Maximum number of line search iterations.
    rtol : float, optional
        Relative tolerance for the objective function. If the relative difference
        between the current and previous value of the objective function is below
        ``rtol``, then the minimization is considered as converged.
    stopping_criteria : Condition, Callable or None, optional
        Additional stopping condition that will make the Gauss-Newton iterations to
        finish. When a condition is passed, the Gauss-Newton iterations will finish if
        the condition is met, the Gauss-Newton converges (relative difference below
        ``rtol``), or if maximum number of iterations are reached.
    cg_kwargs : dict or None, optional
        Dictionary with extra arguments passed to the :func:`scipy.sparse.linalg.cg`
        function.
    """

    def __init__(
        self,
        *,
        maxiter: int = 100,
        maxiter_line_search: int = 10,
        rtol=1e-5,
        stopping_criteria: Condition | Callable[[Model], bool] | None = None,
        cg_kwargs: dict[str, Any] | None = None,
    ):
        self.maxiter = maxiter
        self.maxiter_line_search = maxiter_line_search
        self.rtol = rtol
        self.stopping_criteria = stopping_criteria
        self.cg_kwargs = cg_kwargs if cg_kwargs is not None else {}

    def __call__(
        self,
        objective: Objective,
        initial_model: Model,
        preconditioner: Preconditioner
        | Callable[[Model], Preconditioner]
        | None = None,
    ) -> Generator[Model]:
        """
        Create iterator over Gauss-Newton minimization.
        """
        # Define a static preconditioner for all Gauss-Newton iterations
        cg_kwargs = self.cg_kwargs.copy()

        if preconditioner is not None:
            if "M" in self.cg_kwargs:
                msg = "Cannot simultanously pass `preconditioner` and `M`."
                raise ValueError(msg)
            preconditioner = (
                preconditioner
                if not callable(preconditioner)
                else preconditioner(initial_model)
            )
            cg_kwargs["M"] = preconditioner

        # Perform Gauss-Newton iterations
        iteration = 0
        phi_prev_value = np.inf  # value of the objective function on previous model
        model = initial_model.copy()

        # Yield initial model, so the generator is never empty
        yield model

        # Apply Gauss-Newton iterations
        while True:
            # Stop if reached max number of iterations
            if iteration >= self.maxiter:
                get_logger().info(
                    "⚠️ Reached maximum number of Gauss-Newton iterations "
                    f"({self.maxiter})."
                )
                break

            # Check for convergence
            phi_value = objective(model)
            if (
                not np.isinf(phi_prev_value)
                and np.abs(phi_value - phi_prev_value) <= phi_prev_value * self.rtol
            ):
                break

            # Check for stopping criteria
            if self.stopping_criteria is not None and self.stopping_criteria(model):
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

            # Perform line search
            alpha, n_ls_iters = backtracking_line_search(
                objective,
                model,
                search_direction,
                phi_value=phi_value,
                phi_gradient=gradient,
                maxiter=self.maxiter_line_search,
            )
            if alpha is None:
                msg = (
                    "Couldn't find a valid alpha, obtained None. "
                    f"Ran {n_ls_iters} iterations."
                )
                raise RuntimeError(msg)

            # Perform model step
            model += alpha * search_direction

            # Update cached values and iteration counter
            phi_prev_value = phi_value
            iteration += 1

            # Yield inverted model for the current Gauss-Newon iteration
            yield model
