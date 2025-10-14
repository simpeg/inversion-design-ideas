"""
Classes and functions to build preconditioners.
"""

import numpy as np
import numpy.typing as npt
from scipy.sparse import diags_array, sparray

from .base import Objective


class JacobiPreconditioner:
    """
    Jacobi preconditioner for a given objective function.

    Use this class to define a dynamic Jacobi preconditioner from an objective function.
    This class is a callable that will update the preconditioner for the given model
    each time it gets called. Use this class if you want to update the preconditioner
    on every iteration of the `Inversion`.

    Parameters
    ----------
    objective_function : Objective
        Objective function for which the Jacobi preconditioner will be built.

    See Also
    --------
    get_jacobi_preconditioner
    """

    def __init__(self, objective_function: Objective):
        self.objective_function = objective_function

    def __call__(self, model: npt.NDArray[np.float64]) -> sparray:
        """
        Generate a Jacobi preconditioner as a sparse diagonal array for a given model.

        Parameters
        ----------
        model : (n_params) array
            Model that will be used to build the Jacobi preconditioner from the
            ``objective_function``.

        Returns
        -------
        dia_array
        """
        return get_jacobi_preconditioner(self.objective_function, model)


def get_jacobi_preconditioner(
    objective_function: Objective, model: npt.NDArray[np.float64]
):
    r"""
    Obtain a Jacobi preconditioner from an objective function.

    Parameters
    ----------
    objective_function : Objective
        Objective function from which the preconditioner will be built.
    model : (n_params) array
        Model used to build the preconditioner.

    Returns
    -------
    diag_array
        Preconditioner as a sparse diagonal array.

    Notes
    -----
    Given an objective function :math:`\phi(\mathbf{m})`, this function builds the
    Jacobi preconditioner :math:`\mathbf{P}(\mathbf{m})` as the inverse of the diagonal
    of the Hessian of :math:`\phi(\mathbf{m})`:

    .. math::

        \mathbf{P}(\mathbf{m}) = \text{diag}[ \bar{\bar{\nabla}} \phi(\mathbf{m}) ]^{-1}

    where :math:`\bar{\bar{\nabla}} \phi(\mathbf{m})` is the Hessian of
    :math:`\phi(\mathbf{m})`.
    """
    hessian_diag = objective_function.hessian_diagonal(model)

    # Compute inverse only for non-zero elements
    zeros = hessian_diag == 0.0
    hessian_diag[~zeros] **= -1

    return diags_array(hessian_diag)
