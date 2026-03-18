"""
Classes and functions to build preconditioners.
"""

import numpy as np
import numpy.typing as npt
from scipy.sparse import diags_array, eye_array
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from .base import Objective
from .operators import Identity
from .typing import Model, SparseArray


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

    def __call__(self, model: Model) -> SparseArray:
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


def get_jacobi_preconditioner(objective_function: Objective, model: Model):
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


class BFGSPreconditioner:
    r"""
    BFGS Preconditioner.

    Use this class to define a dynamic BFGS preconditioner from an objective function.

    .. important::

        The preconditioner will get updated every time it gets called.
        Don't use this class on multiple

    Parameters
    ----------
    objective_function : Objective
        Objective function for which the Jacobi preconditioner will be built.
    initial_preconditioner : array or sparray or None, optional
        Square matrix that will be used as the initial estimate of the inverse of the
        Hessian of the ``objective_function``.
        If None, the identity matrix will be used.

    Notes
    -----
    The preconditioner is built using the BFGS update equation (Nocedal & Wright, 1999,
    eq. 8.16):

    .. math::

        H_{k+1} = (I - \rho_k s_k y_k^T) H_k (I - \rho_k y_k s_k^T) + \rho s_k s_k^T,

    where

    .. math::

        \rho_k = \frac{1}{y_k^T s_k},

    .. math::

        \y_k = \nabla \phi(\mathbf{m}_{k+1}) - \nabla \phi(\mathbf{m}_k),

    .. math::

        \s_k = m_{k+1} - m_k,

    and :math:`\phi(.)` is the objective function that will be inverted.

    By default, the :math:`H_0` is set as the identity matrix. Alternatively, the user
    can set an initial value for it through the ``initial_preconditioner`` argument.

    References
    ----------
    Nocedal, J., & Wright, S. J. (1999). Numerical optimization. Springer.
    """

    def __init__(
        self,
        objective_function: Objective,
        initial_preconditioner: (
            npt.NDArray[np.float64] | SparseArray | LinearOperator | None
        ) = None,
    ):
        self.objective_function = objective_function
        if initial_preconditioner is None:
            initial_preconditioner = eye_array(self.objective_function.n_params)
        self.initial_preconditioner = initial_preconditioner
        self._h = aslinearoperator(initial_preconditioner)

    @property
    def h(self) -> LinearOperator:
        """
        Current preconditioner matrix :math:`H_k`.
        """
        return self._h

    @property
    def initialized(self) -> bool:
        """
        Whether the BFGS preconditioner has been initialized.
        """
        return getattr(self, "_initialized", False)

    def __call__(self, model: Model) -> LinearOperator:
        """
        Update the BFGS matrix and return the preconditioner.

        Parameters
        ----------
        model : (n_params) array
            Model that will be used to update the BFGS matrix.

        Returns
        -------
        array or SparseArray
        """
        # Compute gradient with passed model. Make model and gradient vertical vectors.
        new_model = model[:, None]
        new_gradient = self.objective_function.gradient(model)[:, None]

        if not self.initialized:
            # Initialize the preconditioner for further calls.
            self._initialized = True
            self._model_k = new_model
            self._gradient_k = new_gradient
        else:
            s_k = new_model - self._model_k
            y_k = new_gradient - self._gradient_k
            (rho_k,) = (
                1 / (y_k.T @ s_k).ravel()
            )  # extract the single element from the 2d array
            s_k, y_k = aslinearoperator(y_k), aslinearoperator(s_k)

            eye = Identity(self.objective_function.n_params)
            self._h = (eye - rho_k * s_k @ y_k.T) @ self.h @ (
                eye - rho_k * y_k @ s_k.T
            ) + rho_k * s_k @ s_k.T

            # Cache model and gradient
            self._model_k = new_model
            self._gradient_k = new_gradient

        return self.h
