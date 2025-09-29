"""
Regularization classes.
"""
from collections.abc import Iterator
from copy import copy

import numpy as np
import numpy.typing as npt
from scipy.sparse import dia_array, diags_array
import simpeg
import discretize

from .base import Objective


class TikhonovZero(Objective):
    """
    Tikhonov zero-th order regularization.
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
        self.set_name("s")

    def __call__(self, model) -> float:
        model_diff = model - self.reference_model
        weights_matrix = self.weights_matrix
        return model_diff.T @ weights_matrix.T @ weights_matrix @ model_diff

    def gradient(self, model):
        """
        Gradient vector.
        """
        model_diff = model - self.reference_model
        weights_matrix = self.weights_matrix
        return 2 * weights_matrix.T @ weights_matrix @ model_diff

    def hessian(self, model):  # noqa: ARG002
        """
        Hessian matrix.
        """
        weights_matrix = self.weights_matrix
        return 2 * weights_matrix.T @ weights_matrix

    def hessian_diagonal(self, model) -> npt.NDArray[np.float64]:
        """
        Diagonal of the Hessian.
        """
        return self.hessian(model).diagonal()

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
        Diagonal matrix with the regularization weights.
        """
        if isinstance(self.weights, np.ndarray):
            weights_array = self.weights
        elif isinstance(self.weights, dict):
            weights_array = _prod_arrays(iter(self.weights.values()))
        else:
            msg = f"Invalid weights of type '{type(self.weights)}'."
            raise TypeError(msg)
        return diags_array(weights_array)


class Smoothness(Objective):
    """
    Smoothness regularization.
    """

    def __init__(
        self,
        mesh: discretize.base.BaseMesh,
        active_cells: int,
        orientation: str,
        cell_weights: npt.NDArray | dict[str, npt.NDArray] | None = None,
        reference_model=None,
    ):
        self.mesh = mesh
        self.active_cells = active_cells
        self.orientation = orientation

        if cell_weights is None:
            cell_weights = np.ones(self.n_params, dtype=np.float64)
        self.cell_weights = cell_weights  # assign the weights through the setter

        self.reference_model = (
            reference_model
            if reference_model is not None
            else np.zeros(self.n_params, dtype=np.float64)
        )
        self.set_name(orientation)

    @property
    def n_params(self) -> int:
        return np.sum(self.active_cells)

    def __call__(self, model) -> float:
        model_diff = model - self.reference_model
        weights_matrix = self.weights_matrix
        cell_gradient = self._cell_gradient
        return (
            model_diff.T
            @ cell_gradient.T
            @ weights_matrix.T
            @ weights_matrix
            @ cell_gradient
            @ model_diff
        )

    def gradient(self, model):
        """
        Gradient vector.
        """
        model_diff = model - self.reference_model
        weights_matrix = self.weights_matrix
        cell_gradient = self._cell_gradient
        return (
            2
            * cell_gradient.T
            @ weights_matrix.T
            @ weights_matrix
            @ cell_gradient
            @ model_diff
        )

    def hessian(self, model):  # noqa: ARG002
        """
        Hessian matrix.
        """
        weights_matrix = self.weights_matrix
        cell_gradient = self._cell_gradient
        return 2 * cell_gradient.T @ weights_matrix.T @ weights_matrix @ cell_gradient

    def hessian_diagonal(self, model) -> npt.NDArray[np.float64]:
        """
        Diagonal of the Hessian.
        """
        return self.hessian(model).diagonal()

    @property
    def cell_weights(
        self,
    ) -> npt.NDArray[np.float64] | dict[str, npt.NDArray[np.float64]]:
        """
        Regularization weights on cells.
        """
        return self._cell_weights

    @cell_weights.setter
    def cell_weights(
        self, value: npt.NDArray[np.float64] | dict[str, npt.NDArray[np.float64]]
    ):
        """
        Setter for weights on cells.
        """
        if not isinstance(value, np.ndarray | dict):
            msg = (
                f"Invalid weights of type {type(value)}. "
                "It must be an array or a dictionary."
            )
            raise TypeError(msg)
        self._cell_weights = value

    @property
    def weights_matrix(self) -> dia_array:
        """
        Diagonal matrix with the regularization weights on faces.
        """
        if isinstance(self.cell_weights, np.ndarray):
            cell_weights = self.cell_weights
        elif isinstance(self.cell_weights, dict):
            cell_weights = _prod_arrays(iter(self.cell_weights.values()))
        else:
            msg = f"Invalid weights of type '{type(self.cell_weights)}'."
            raise TypeError(msg)

        avg_cell_to_face = getattr(
            self._regularization_mesh, f"aveCC2F{self.orientation}"
        )
        return diags_array(avg_cell_to_face @ cell_weights)

    @property
    def _cell_gradient(self):
        """Return the cell gradient matrix operator."""
        if not hasattr(self, "__cell_gradient"):
            self.__cell_gradient = getattr(
                self._regularization_mesh, f"cell_gradient_{self.orientation}"
            )
        return self.__cell_gradient

    @property
    def _regularization_mesh(self):
        """Return a :class:`simpeg.RegularizationMesh`."""
        # TODO: would be nice to simplify this, don't quite like the idea of
        # regularization meshes. Even if we keep them, I think they should be private.
        if not hasattr(self, "_regmesh"):
            self._regmesh = simpeg.regularization.RegularizationMesh(
                self.mesh, self.active_cells
            )
        return self._regmesh


def _prod_arrays(arrays: Iterator[npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    """
    Compute product of arrays within an iterator.

    Parameters
    ----------
    arrays : Iterator
        Iterator with arrays.
    """
    if not arrays:
        msg = "Invalid empty 'arrays' array when summing."
        raise ValueError(msg)

    result = copy(next(arrays))
    for array in arrays:
        result *= array
    return result
