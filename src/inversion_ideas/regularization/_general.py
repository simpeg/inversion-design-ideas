"""
General purpose regularization classes.
"""

import numpy as np
import numpy.typing as npt
from scipy.sparse import dia_array, diags_array

from .._utils import prod_arrays
from ..base import Objective
from ..typing import Model


class TikhonovZero(Objective):
    r"""
    Tikhonov zero-th order regularization.

    Parameters
    ----------
    n_params : int
        Number of elements in the ``model`` array.
    weights : (n_params) array or dict of (n_params) arrays or None, optional
        Array with regularization weights.
        For multiple weights, pass a dictionary where keys are strings and values are
        the different weights arrays.
        If None, no weights are going to be used.
    reference_model : (n_params) array or None, optional
        Array with values for the reference model.

    Notes
    -----
    Implement a Tikhonov zero-th order regularization as follows:

    .. math::

        \phi(\mathbf{m})
        = \sum\limits_{i=1}^M w_i |m_i - m_i^\text{ref}|^2
        = \lVert \mathbf{W} (\mathbf{m} - \mathbf{m}^\text{ref}) \rVert^2

    where :math:`\mathbf{W} = [\sqrt{w_1}, \dots, \sqrt{w_M}]` are the square roots of
    the regularization weights,
    :math:`\mathbf{m} = [m_1, \dots, m_M]` and :math:`\mathbf{m}^\text{ref}
    = [m_1^\text{ref}, \dots, m_M^\text{ref}]` are the model and reference model
    vectors, respectively.
    """

    def __init__(
        self,
        n_params: int,
        weights: npt.NDArray | dict[str, npt.NDArray] | None = None,
        reference_model=None,
    ):
        self._n_params = n_params

        if weights is None:
            weights = np.ones(n_params, dtype=np.float64)
        self.weights = weights

        self.reference_model = (
            reference_model
            if reference_model is not None
            else np.zeros(n_params, dtype=np.float64)
        )
        self.set_name("0")

    def __call__(self, model: Model) -> float:
        """
        Evaluate the regularization on a given model.

        Parameters
        ----------
        model : (n_params) array
            Array with model values.
        """
        model_diff = model - self.reference_model
        weights_matrix = self.weights_matrix
        return model_diff.T @ weights_matrix.T @ weights_matrix @ model_diff

    def gradient(self, model: Model):
        """
        Gradient vector.

        Parameters
        ----------
        model : (n_params) array
            Array with model values.
        """
        model_diff = model - self.reference_model
        weights_matrix = self.weights_matrix
        return 2 * weights_matrix.T @ weights_matrix @ model_diff

    def hessian(self, model: Model):  # noqa: ARG002
        """
        Hessian matrix.

        Parameters
        ----------
        model : (n_params) array
            Array with model values.
        """
        weights_matrix = self.weights_matrix
        return 2 * weights_matrix.T @ weights_matrix

    @property
    def n_params(self):
        """
        Number of model parameters.
        """
        return self._n_params

    @property
    def weights(self) -> npt.NDArray[np.float64] | dict[str, npt.NDArray[np.float64]]:
        """
        Regularization weights.
        """
        return self._weights

    @weights.setter
    def weights(
        self, value: npt.NDArray[np.float64] | dict[str, npt.NDArray[np.float64]]
    ):
        """
        Setter for weights.
        """
        if not isinstance(value, np.ndarray | dict):
            msg = (
                f"Invalid weights of type {type(value)}. "
                "It must be an array or a dictionary."
            )
            raise TypeError(msg)
        self._weights = value

    @property
    def weights_matrix(self) -> dia_array:
        """
        Diagonal matrix with the square root of the regularization weights.
        """
        if isinstance(self.weights, np.ndarray):
            weights_array = self.weights
        elif isinstance(self.weights, dict):
            weights_array = prod_arrays(iter(self.weights.values()))
        else:
            msg = f"Invalid weights of type '{type(self.weights)}'."
            raise TypeError(msg)
        return diags_array(np.sqrt(weights_array))


class TikhonovFirst(Objective):
    r"""
    Tikhonov first order regularization.

    Parameters
    ----------
    n_params : int
        Number of elements in the ``model`` array.

    Notes
    -----
    Implement a Tikhonov first order regularization as follows:

    .. math::

        \phi(\mathbf{m})
        = \sum\limits_{i=1}^{M-1} |m_{i+1} - m_{i}|^2
        = \lVert \mathbf{R} \mathbf{m} \rVert^2
        = \mathbf{m}^\text{T} \mathbf{R}^\text{T} \mathbf{R} \mathbf{m}

    where :math:`\mathbf{m} = [m_1, \dots, m_M]` is the model, and
    :math:`\mathbf{R}` is the matrix of finite differences:

    .. math::

        \mathbf{R} =
        \begin{bmatrix}
        1      & -1      &        &        &        & 0      \\
               &  1      & -1     &        &        &        \\
               &         & \ddots & \ddots &        &        \\
               &         &        & 1      & -1     &        \\
        0      &         &        &        & -1     & 1      \\
        \end{bmatrix}.

    """

    def __init__(self, n_params: int):
        self._n_params = n_params
        self.set_name("1")

    @property
    def n_params(self):
        """
        Number of model parameters.
        """
        return self._n_params

    def __call__(self, model: Model) -> float:
        """
        Evaluate the regularization on a given model.

        Parameters
        ----------
        model : (n_params) array
            Array with model values.

        Returns
        -------
        float
        """
        r_matrix = self.finite_differences_matrix
        return model.T @ r_matrix.T @ r_matrix @ model

    def gradient(self, model: Model) -> npt.NDArray[np.float64]:
        """
        Gradient vector.

        Parameters
        ----------
        model : (n_params) array
            Array with model values.

        Returns
        -------
        (n_params) array
        """
        r_matrix = self.finite_differences_matrix
        return 2 * r_matrix.T @ r_matrix @ model

    def hessian(self, model: Model):  # noqa: ARG002
        """
        Hessian matrix.

        Parameters
        ----------
        model : (n_params) array
            Array with model values.

        Returns
        -------
        (n_params, n_params) sparray
        """
        r_matrix = self.finite_differences_matrix
        return 2 * r_matrix.T @ r_matrix

    @property
    def finite_differences_matrix(self):
        if not hasattr(self, "_finite_differences_matrix"):
            diagonals = [
                np.ones(self.n_params - 1),
                -np.ones(self.n_params - 1),
            ]
            offsets = [0, 1]
            shape = (self.n_params - 1, self.n_params)
            self._finite_differences_matrix = diags_array(
                diagonals, offsets=offsets, shape=shape
            )
        return self._finite_differences_matrix
