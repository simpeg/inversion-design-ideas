"""
Classes and functions to build preconditioners.
"""

import warnings

import numpy as np
import numpy.typing as npt
from scipy.sparse import dia_array, diags_array
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from .base import Objective
from .operators import Identity
from .typing import Model, SparseArray


class JacobiPreconditioner(LinearOperator):
    """
    Jacobi preconditioner for a given objective function.

    Use this class to define a dynamic Jacobi preconditioner from an objective function.
    This class implements the ``update`` method that can be used by minimizers to
    dynamically update the preconditioner.

    Parameters
    ----------
    objective_function : Objective
        Objective function for which the Jacobi preconditioner will be built.
    dtype : dtype, optional
        Data type of the matrix.

    See Also
    --------
    get_jacobi_preconditioner
    """

    def __init__(self, objective_function: Objective, dtype=np.float64):
        self.objective_function = objective_function
        n = self.objective_function.n_params
        super().__init__(shape=(n, n), dtype=dtype)

    @property
    def A(self) -> dia_array:
        if not hasattr(self, "_preconditioner"):
            msg = (
                f"Preconditioner {self} doesn't have a `A` attribute "
                "since it hasn't been initialized yet."
            )
            raise AttributeError(msg)
        return self._preconditioner

    def update(self, model: Model):
        """
        Update the preconditioner.
        """
        self._preconditioner = get_jacobi_preconditioner(self.objective_function, model)

    def _matvec(self, x):
        return self.A @ x

    def _rmatvec(self, x):
        return self.A.T @ x


def get_jacobi_preconditioner(objective_function: Objective, model: Model) -> dia_array:
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
    hessian_diag = objective_function.hessian_diagonal(model).diagonal()

    # Compute inverse only for non-zero elements
    zeros = hessian_diag == 0.0
    preconditioner = hessian_diag.copy()
    preconditioner[~zeros] **= -1

    return diags_array(preconditioner)


class BFGSPreconditioner(LinearOperator):
    r"""
    BFGS Preconditioner.

    Use this class to define a dynamic BFGS preconditioner from an objective function.
    This class implements the ``update`` method that can be used by minimizers to
    dynamically update the preconditioner.

    Parameters
    ----------
    objective_function : Objective
        Objective function for which the Jacobi preconditioner will be built.
    initial_matrix : array or sparray or LinearOperator or None, optional
        Square matrix that will be used as the initial estimate of the inverse of the
        Hessian of the ``objective_function``.
        If None, the identity matrix will be used.
    dtype : dtype, optional
        Data type of the matrix.

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

    and :math:`\phi(\cdot)` is the objective function that will be inverted.

    By default, the :math:`H_0` is set as the identity matrix. Alternatively, the user
    can set an initial value for it through the ``initial_preconditioner`` argument.

    References
    ----------
    Nocedal, J., & Wright, S. J. (1999). Numerical optimization. Springer.
    """

    def __init__(
        self,
        objective_function: Objective,
        initial_matrix: (
            npt.NDArray[np.float64] | SparseArray | LinearOperator | None
        ) = None,
        dtype=np.float64,
    ):
        n = objective_function.n_params
        self.objective_function = objective_function
        super().__init__(shape=(n, n), dtype=dtype)
        self.initial_matrix = (
            initial_matrix if initial_matrix is not None else Identity(n)
        )
        self._index = None

    @property
    def index(self) -> int | None:
        """
        Current index of the BFGS update algorithm.
        """
        return self._index

    @property
    def matrix(self) -> LinearOperator:
        """
        Current preconditioner matrix :math:`H_k`.
        """
        if not hasattr(self, "_matrix"):
            return aslinearoperator(self.initial_matrix)
        return self._matrix

    def _matvec(self, x):
        return self.matrix @ x

    def _rmatvec(self, x):
        return self.matrix.T @ x

    def reset(self):
        """
        Reset the BFGS preconditioner.

        Ditch the current preconditioner matrix estimation and start over with the
        initial matrix. Resets the ``index`` to None.
        """
        self._index = None
        del self._matrix
        del self._model_k
        del self._gradient_k

    def update(self, model: Model):
        """
        Update the BFGS preconditioner.

        Parameters
        ----------
        model : (n_params,) array
            Model array.
        """
        # Get new model and gradient (cast model and gradient as vertical vectors).
        new_model = model[:, None].copy()
        new_gradient = self.objective_function.gradient(model)[:, None]

        if self.index is None:
            # Define the first matrix as a LinearOperator containing the initial matrix
            self._matrix = aslinearoperator(self.initial_matrix)
            self._index = 0
        else:
            # Compute variables to update the preconditioner
            s_k = new_model - self._model_k
            y_k = new_gradient - self._gradient_k
            (y_dot_s,) = (
                y_k.T @ s_k
            ).ravel()  # extract the single element from the 2d array
            rho_k = 1 / y_dot_s
            s_k, y_k = aslinearoperator(y_k), aslinearoperator(s_k)
            eye = Identity(self.objective_function.n_params)

            if y_dot_s <= 0:
                msg = (
                    "Found `y.T @ s <= 0` while updating BFGS preconditioner. "
                    "Skipping updating step."
                    "The skipping process is still in experimental phase, "
                    "please open an issue if you received this warning!"
                )
                warnings.warn(msg, stacklevel=2)

                # Cache model and gradient
                self._model_k = new_model
                self._gradient_k = new_gradient
                self._index += 1
                return

            # Update the preconditioner
            left = eye - rho_k * (s_k @ y_k.T)
            right = eye - rho_k * (y_k @ s_k.T)
            self._matrix = left @ self._matrix @ right + rho_k * (s_k @ s_k.T)

            self._index += 1

        # Cache model and gradient
        self._model_k = new_model
        self._gradient_k = new_gradient
