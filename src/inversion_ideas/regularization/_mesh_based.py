"""
Regularization classes for mesh-based inversion problems.
"""

import discretize
import numpy as np
import numpy.typing as npt
import simpeg
from scipy.sparse import dia_array, diags_array

from .._utils import prod_arrays
from ..base import Objective


class Smallness(Objective):
    r"""
    Smallness regularization.

    Regularize a weighted norm of model values.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
        Mesh to use in the regularization.
    active_cells : (n_params) array
        Array full of bools that indicate the active cells in the mesh.
    cell_weights : (n_params) array or dict of (n_params) arrays or None, optional
        Array with cell weights.
        For multiple cell weights, pass a dictionary where keys are strings and values
        are the different weights arrays.
        If None, no cell weights are going to be used.
    reference_model : (n_params) array or None, optional
        Array with values for the reference model.

    Notes
    -----
    Implement a discretized version of the smallness regularization defined as follows:

    .. math::

        \phi_s(\mathbf{m}) =
        \int_\Omega
        w(\mathbf{r})
        |m(\mathbf{r}) - m^\text{ref}(\mathbf{r})|^2
        \text{d}\mathbf{r},

    where :math:`w(\mathbf{r})` is the value of weights on the point :math:`\mathbf{r}`,
    :math:`m` and :math:`m^\text{ref}` are the model and reference model, respectively.

    When discretizing it into the ``mesh``, the smallness regularization can be
    expressed by:

    .. math::

        \phi_s(\mathbf{m})
        =
        \sum\limits_{i=0}^M
        w_i
        V_i
        |m_i - m_i^\text{ref}|^2
        =
        \lVert
        \mathbf{W}
        \mathbf{V}
        (\mathbf{m} - \mathbf{m}^\text{ref})
        \rVert^2,

    where :math:`\mathbf{W} = [\sqrt{w_1}, \dots, \sqrt{w_M}]` are the square roots of
    the cell weights,
    :math:`\mathbf{V} = [\sqrt{V_1}, \dots, \sqrt{V_M}]`
    are the square root of cell volumes,
    :math:`\mathbf{m} = [m_1, \dots, m_M]` and
    :math:`\mathbf{m}^\text{ref} = [m_1^\text{ref}, \dots, m_M^\text{ref}]`
    are the model and reference model vectors, respectively.
    """

    def __init__(
        self,
        mesh: discretize.base.BaseMesh,
        active_cells: int,
        cell_weights: npt.NDArray | dict[str, npt.NDArray] | None = None,
        reference_model=None,
    ):
        self.mesh = mesh
        self.active_cells = active_cells

        if cell_weights is None:
            cell_weights = np.ones(self.n_params, dtype=np.float64)
        self.cell_weights = cell_weights  # assign the weights through the setter

        self.reference_model = (
            reference_model
            if reference_model is not None
            else np.zeros(self.n_params, dtype=np.float64)
        )
        self.set_name("s")

    @property
    def n_params(self) -> int:
        return np.sum(self.active_cells)

    def __call__(self, model) -> float:
        """
        Evaluate the regularization on a given model.

        Parameters
        ----------
        model : (n_params) array
            Array with model values.
        """
        model_diff = model - self.reference_model
        weights_matrix = self.weights_matrix
        cell_volumes_sqrt = self._volumes_sqrt_matrix
        return (
            model_diff.T
            @ cell_volumes_sqrt.T
            @ weights_matrix.T
            @ weights_matrix
            @ cell_volumes_sqrt
            @ model_diff
        )

    def gradient(self, model):
        """
        Gradient vector.

        Parameters
        ----------
        model : (n_params) array
            Array with model values.
        """
        model_diff = model - self.reference_model
        weights_matrix = self.weights_matrix
        cell_volumes_sqrt = self._volumes_sqrt_matrix
        return (
            2
            * cell_volumes_sqrt.T
            @ weights_matrix.T
            @ weights_matrix
            @ cell_volumes_sqrt
            @ model_diff
        )

    def hessian(self, model):  # noqa: ARG002
        """
        Hessian matrix.

        Parameters
        ----------
        model : (n_params) array
            Array with model values.
        """
        weights_matrix = self.weights_matrix
        cell_volumes_sqrt = self._volumes_sqrt_matrix
        return (
            2
            * cell_volumes_sqrt.T
            @ weights_matrix.T
            @ weights_matrix
            @ cell_volumes_sqrt
        )

    def hessian_diagonal(self, model) -> npt.NDArray[np.float64]:
        """
        Diagonal of the Hessian.

        Parameters
        ----------
        model : (n_params) array
            Array with model values.
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
        Diagonal matrix with the square root of regularization weights on faces.
        """
        if isinstance(self.cell_weights, np.ndarray):
            cell_weights = self.cell_weights
        elif isinstance(self.cell_weights, dict):
            cell_weights = prod_arrays(iter(self.cell_weights.values()))
        else:
            msg = f"Invalid weights of type '{type(self.cell_weights)}'."
            raise TypeError(msg)
        return diags_array(np.sqrt(cell_weights))

    @property
    def _volumes_sqrt_matrix(self) -> dia_array:
        """
        Diagonal matrix with the square root of cell volumes.
        """
        cell_volumes = self.mesh.cell_volumes[self.active_cells]
        return diags_array(np.sqrt(cell_volumes))


class Smoothness(Objective):
    r"""
    Smoothness regularization.

    Regularize a weighted norm of a spatial derivative of the model.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
        Mesh to use in the regularization.
    active_cells : (n_params) array
        Array full of bools that indicate the active cells in the mesh.
    direction : {"x", "y", "z"}
        Direction of the spatial derivative.
    cell_weights : (n_params) array or dict of (n_params) arrays or None, optional
        Array with cell weights.
        For multiple cell weights, pass a dictionary where keys are strings and values
        are the different weights arrays.
        If None, no cell weights are going to be used.
    reference_model : (n_params) array or None, optional
        Array with values for the reference model.

    Notes
    -----
    Implement a discretized version of the smoothness regularization defined as follows,
    assuming that the ``direction`` of the derivative is along :math:`x`:

    .. math::

        \phi_x(\mathbf{m}) =
        \int_\Omega
        w(\mathbf{r})
        \lvert
        \frac{\partial m}{\partial x}
        -
        \frac{\partial m^\text{ref}}{\partial x}
        \rvert^2
        \text{d}\mathbf{r},

    where :math:`w(\mathbf{r})` is the value of weights on the point :math:`\mathbf{r}`,
    :math:`m` and :math:`m^\text{ref}` are the model and reference model, respectively.

    When discretizing it into the ``mesh``, the smallness regularization can be
    expressed by:

    .. math::

        \phi_s(\mathbf{m})
        =
        \lVert
        \mathbf{W}^f
        \mathbf{V}^f
        \mathbf{G}_x
        (\mathbf{m} - \mathbf{m}^\text{ref})
        \rVert^2,

    where :math:`\mathbf{W}^f` are the square root of cell weights averaged on faces,
    :math:`\mathbf{V}^f`
    are the square root of cell volumes averaged on faces,
    :math:`\mathbf{G}_x` is the partial cell gradient operator along the :math:`x`
    direction,
    :math:`\mathbf{m} = [m_1, \dots, m_M]` and
    :math:`\mathbf{m}^\text{ref} = [m_1^\text{ref}, \dots, m_M^\text{ref}]`
    are the model and reference model vectors, respectively.

    .. important::

        After applying the :math:`\mathbf{G}_x` to the model, the partial derivatives
        are located on the center of the faces. For this reason we need to average the
        cell volumes and the cell weights to faces before using them in the
        regularization.
    """

    def __init__(
        self,
        mesh: discretize.base.BaseMesh,
        active_cells: int,
        direction: str,
        cell_weights: npt.NDArray | dict[str, npt.NDArray] | None = None,
        reference_model=None,
    ):
        self.mesh = mesh
        self.active_cells = active_cells
        self.direction = direction

        if cell_weights is None:
            cell_weights = np.ones(self.n_params, dtype=np.float64)
        self.cell_weights = cell_weights  # assign the weights through the setter

        self.reference_model = (
            reference_model
            if reference_model is not None
            else np.zeros(self.n_params, dtype=np.float64)
        )
        self.set_name(direction)

    @property
    def n_params(self) -> int:
        return np.sum(self.active_cells)

    def __call__(self, model) -> float:
        """
        Evaluate the regularization on a given model.

        Parameters
        ----------
        model : (n_params) array
            Array with model values.
        """
        model_diff = model - self.reference_model
        weights_matrix = self.weights_matrix
        cell_volumes_sqrt = self._volumes_sqrt_matrix
        cell_gradient = self._cell_gradient
        return (
            model_diff.T
            @ cell_gradient.T
            @ cell_volumes_sqrt.T
            @ weights_matrix.T
            @ weights_matrix
            @ cell_volumes_sqrt
            @ cell_gradient
            @ model_diff
        )

    def gradient(self, model):
        """
        Gradient vector.

        Parameters
        ----------
        model : (n_params) array
            Array with model values.
        """
        model_diff = model - self.reference_model
        weights_matrix = self.weights_matrix
        cell_volumes_sqrt = self._volumes_sqrt_matrix
        cell_gradient = self._cell_gradient
        return (
            2
            * cell_gradient.T
            @ cell_volumes_sqrt.T
            @ weights_matrix.T
            @ weights_matrix
            @ cell_volumes_sqrt
            @ cell_gradient
            @ model_diff
        )

    def hessian(self, model):  # noqa: ARG002
        """
        Hessian matrix.

        Parameters
        ----------
        model : (n_params) array
            Array with model values.
        """
        weights_matrix = self.weights_matrix
        cell_gradient = self._cell_gradient
        cell_volumes_sqrt = self._volumes_sqrt_matrix
        return (
            2
            * cell_gradient.T
            @ cell_volumes_sqrt.T
            @ weights_matrix.T
            @ weights_matrix
            @ cell_volumes_sqrt
            @ cell_gradient
        )

    def hessian_diagonal(self, model) -> npt.NDArray[np.float64]:
        """
        Diagonal of the Hessian.

        Parameters
        ----------
        model : (n_params) array
            Array with model values.
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
        Diagonal matrix with the square root of cell weights averaged on faces.
        """
        if isinstance(self.cell_weights, np.ndarray):
            cell_weights = self.cell_weights
        elif isinstance(self.cell_weights, dict):
            cell_weights = prod_arrays(iter(self.cell_weights.values()))
        else:
            msg = f"Invalid weights of type '{type(self.cell_weights)}'."
            raise TypeError(msg)
        return diags_array(self._average_cells_to_faces @ np.sqrt(cell_weights))

    @property
    def _volumes_sqrt_matrix(self) -> dia_array:
        """
        Diagonal matrix with the square root of cell volumes averaged on faces.
        """
        cell_volumes = self.mesh.cell_volumes[self.active_cells]
        return diags_array(self._average_cells_to_faces @ np.sqrt(cell_volumes))

    @property
    def _average_cells_to_faces(self):
        """Sparse matrix to average from cell centers to faces."""
        return getattr(self._regularization_mesh, f"aveCC2F{self.direction}")

    @property
    def _cell_gradient(self):
        """Return the cell gradient matrix operator."""
        if not hasattr(self, "__cell_gradient"):
            self.__cell_gradient = getattr(
                self._regularization_mesh, f"cell_gradient_{self.direction}"
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
