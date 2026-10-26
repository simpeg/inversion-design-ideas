"""
Utility functions for minimizers.
"""

import copy

import numpy as np
import numpy.typing as npt

from ..base import Objective
from ..typing import Model


def backtracking_line_search(
    phi: Objective,
    model: Model,
    search_direction: Model,
    *,
    contraction_factor: float = 0.5,
    c_factor: float = 0.5,
    maxiter: int = 20,
    initial_step_length: float = 1.0,
    phi_value: float | None = None,
    phi_gradient: npt.NDArray[np.float64] | None = None,
) -> tuple[float | None, int]:
    r"""
    Implement the backtracking line search algorithm.

    Parameters
    ----------
    phi : base.Objective
        Objective function to which the line search will be applied.
    model : (n_params) array
        Current model.
    search_direction: (n_params) array
        Vector used as a search direction.
    contraction_factor : float, optional
        Contraction factor for the step length. Must be greater than 0 and lower than 1.
    c_factor : float, optional
        The c factor used in the descent condition.
        Must be greater than 0 and lower than 1.
    maxiter : int, optional
        Maximum number of line search iterations.
    initial_step_length : float, optional
        Initial value for the step length.
    phi_value : float or None, optional
        Precomputed value of ``phi(model)``. If None, it will be computed.
        Pass a value to avoid retriggering a call of ``phi`` in case you've already
        called it.
    phi_gradient : (n_params) array or None, optional
        Precomputed value of ``phi.gradient(model)``. If None, it will be computed.
        Pass a value to avoid retriggering a call of ``phi.gradient`` in case you've
        already called it.

    Returns
    -------
    step_length : float or None
        Alpha for which ``x_new = x0 + alpha * pk``, or None if the line search
        algorithm did not converge.
    n_iterations : int
        Number of line search iterations.

    Notes
    -----
    Given an objective function :math:`\phi` that is being iteratively minimized, a
    current model :math:`\mathbf{m}_k` and a search direction :math:`\mathbf{p}_k`,
    the backtracking line search algorithm offers a way to choose the step length
    :math:`\alpha` that we will take in the direction of :math:`\mathbf{p}_k` to obtain
    the next model :math:`\mathbf{m}_{k+1}`:

    .. math::

        \mathbf{m}_{k+1} = \mathbf{m}_k + \alpha_k \mathbf{p}_k


    Ideally, the step length `\alpha_k` should be choose such as the *sufficient
    decrease* and the *curvature* conditions are met.
    These are also know as the *Wolfe conditions*. Respectively:

    .. math::

        \begin{aligned}
            \phi(\mathbf{m}_k + \alpha_k \mathbf{p}_k)
                &\le
                \phi(\phi(\mathbf{m}_k)
                + c_1 \alpha_k {\nabla\phi(\mathbf{m}_k)}^\text{T} \mathbf{p}_k \\
            {\nabla\phi(\mathbf{m}_k + \alpha_k \mathbf{p}_k)}^\text{T} \mathbf{p}_k
                &\ge
                c_2 {\nabla\phi(\mathbf{m}_k)}^\text{T} \mathbf{p}_k
        \end{aligned}

    with :math:`0 < c_1 < c_2 < 1`.

    The *backtracking line search* approach consists in disposing the curvature
    condition and choosing the candidate step lengths appropriately.

    .. note::

        We would like to avoid enforcing the curvature condition since it involves
        computing the gradient of :math:`\phi` once per step length candidate.
        If computing the gradient of :math:`\phi` for a given model is a costly
        operation, that could make the minimization algorithm to take considerable
        time.

    The backtracking line search can be summarized in:

    1) Choose an initial positive value for the step length. Usually this value is set to one.
    2) Choose values :math:`\in (0, 1)` for the *contraction factor* :math:`\rho` and the factor :math:`c`.
    3) Check if the condition
       :math:`\phi(\mathbf{m}_k + \alpha\mathbf{p}_k) \le \phi(\mathbf{m}_k) + c \alpha {\nabla\phi(\mathbf{m}_k)}^\text{T} \mathbf{p}_k` holds.

       * If it does, then **finish the algorithm**. Use the current value of :math:`\alpha` as the step length.
       * If it doesn't, **continue to next step**.

    4) Update the step length as :math:`\alpha \leftarrow \rho\alpha`.


    References
    ----------
    Nocedal, J., & Wright, S. J. (1999). Numerical optimization. Springer. Page 41.
    """
    for argument, value in {
        "contraction_factor": contraction_factor,
        "c_factor": c_factor,
    }.items():
        if not (0 < value < 1):
            msg = (
                f"Invalid {argument} value of '{value}'. "
                "It must be larger than zero and lower than one."
            )
            raise ValueError(msg)

    phi_value = phi_value if phi_value is not None else phi(model)
    phi_gradient = phi_gradient if phi_gradient is not None else phi.gradient(model)

    def stop_condition(step_length):
        return (
            phi(model + step_length * search_direction)
            <= phi_value + c_factor * step_length * phi_gradient @ search_direction
        )

    step_length = copy.copy(initial_step_length)
    n_iterations = 0
    while not stop_condition(step_length):
        step_length *= contraction_factor
        n_iterations += 1

        if n_iterations >= maxiter:
            return None, n_iterations

    return step_length, n_iterations
