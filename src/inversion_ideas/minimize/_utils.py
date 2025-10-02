"""
Utility functions for minimizers.
"""
import numpy as np
import numpy.typing as npt

from ..base import Objective


def backtracking_line_search(
    phi: Objective,
    model: npt.NDArray[np.float64],
    search_direction: npt.NDArray[np.float64],
    *,
    contraction_factor: float = 0.5,
    c_factor: float = 0.5,
    phi_value: float | None = None,
    phi_gradient: npt.NDArray[np.float64] | None = None,
    maxiter: int = 20,
) -> tuple[float | None, int]:
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
