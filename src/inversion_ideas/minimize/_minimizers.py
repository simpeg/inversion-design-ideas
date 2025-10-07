"""
Minimizer classes.
"""

import warnings
from collections.abc import Callable, Generator
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.sparse.linalg import cg

from ..base import Minimizer, Objective
from ..errors import ConvergenceWarning
from ..typing import Model, Preconditioner
from ..utils import get_logger
from ._utils import backtracking_line_search


class GaussNewtonConjugateGradient(Minimizer):
    def __init__(
        self,
        *,
        maxiter: int = 100,
        maxiter_line_search: int = 10,
        rtol=1e-5,
        cg_kwargs: dict[str, Any] | None = None,
    ):
        self.maxiter = maxiter
        self.maxiter_line_search = maxiter_line_search
        self.rtol = rtol
        self.cg_kwargs = cg_kwargs if cg_kwargs is not None else {}

    def __call__(
        self,
        objective: Objective,
        initial_model: npt.NDArray[np.float64],
        preconditioner: Preconditioner
        | Callable[[Model], Preconditioner]
        | None = None,
    ) -> Generator[npt.NDArray[np.float64]]:
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
                if not isinstance(preconditioner, Callable)
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

            # Check for stopping criteria
            phi_value = objective(model)
            if (
                not np.isinf(phi_prev_value)
                and np.abs(phi_value - phi_prev_value) <= phi_prev_value * self.rtol
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
