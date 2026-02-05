"""
Regularization classes for mesh-based inversion problems.
"""

import discretize
import numpy as np
import numpy.typing as npt
import simpeg
from scipy.sparse import dia_array, diags_array, eye_array

from .._utils import prod_arrays
from ..base import Objective
from ..typing import Model
from ..wires import ModelSlice


class _MeshBasedRegularization(Objective):
    """
    Base class for mesh-based regularizations.

    Implements common methods like ``cell_weights``, the ``n_params`` property.
    """

    active_cells: npt.NDArray[np.bool]

    @property
    def n_params(self) -> int:
        if (model_slice := getattr(self, "model_slice", None)) is not None:
            return model_slice.full_size
        return self.n_active

    @property
    def n_active(self) -> int:
        return int(np.sum(self.active_cells))

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
        if isinstance(value, np.ndarray) and value.size != self.n_active:
            msg = (
                f"Invalid cell_weights array with '{value.size}' elements. "
                f"It must have '{self.n_params}' elements, "
                "equal to the number of active cells."
            )
            raise ValueError(msg)
        if isinstance(value, dict):
            for key, array in value.items():
                if array.size != self.n_active:
                    msg = (
                        f"Invalid cell_weights array '{key}' with "
                        f"'{array.size}' elements. "
                        f"It must have '{self.n_params}' elements, "
                        "equal to the number of active cells."
                    )
                    raise ValueError(msg)
        self._cell_weights = value


class Smallness(_MeshBasedRegularization):
    r"""
    Smallness regularization.

    Regularize a weighted norm of model values.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
        Mesh to use in the regularization.
    active_cells : (n_cells) array or None, optional
        Array full of bools that indicate the active cells in the mesh. It must have the
        same amount of elements as cells in the mesh.
    cell_weights : (n_active) array or dict of (n_active) arrays or None, optional
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
        *,
        active_cells: npt.NDArray[np.bool] | None = None,
        cell_weights: npt.NDArray | dict[str, npt.NDArray] | None = None,
        reference_model: Model | None = None,
        model_slice: ModelSlice | None = None,
    ):
        self.mesh = mesh
        self.active_cells = (
            active_cells
            if active_cells is not None
            else np.ones(self.mesh.n_cells, dtype=bool)
        )
        # assign model_slice after active_cells so n_params is correct during __init__
        self.model_slice = model_slice

        # Assign the cell weights through the setter
        self.cell_weights = (
            cell_weights
            if cell_weights is not None
            else np.ones(self.n_active, dtype=np.float64)
        )

        self.reference_model = (
            reference_model
            if reference_model is not None
            else np.zeros(self.n_params, dtype=np.float64)
        )
        self.set_name("s")

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
        cell_volumes_sqrt = self._volumes_sqrt_matrix
        slicer_matrix = self._slicer_matrix
        return (
            model_diff.T
            @ slicer_matrix.T
            @ cell_volumes_sqrt.T
            @ weights_matrix.T
            @ weights_matrix
            @ cell_volumes_sqrt
            @ slicer_matrix
            @ model_diff
        )

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
        cell_volumes_sqrt = self._volumes_sqrt_matrix
        slicer_matrix = self._slicer_matrix
        return (
            2
            * slicer_matrix.T
            @ cell_volumes_sqrt.T
            @ weights_matrix.T
            @ weights_matrix
            @ cell_volumes_sqrt
            @ slicer_matrix
            @ model_diff
        )

    def hessian(self, model: Model):  # noqa: ARG002
        """
        Hessian matrix.

        Parameters
        ----------
        model : (n_params) array
            Array with model values.
        """
        weights_matrix = self.weights_matrix
        cell_volumes_sqrt = self._volumes_sqrt_matrix
        slicer_matrix = self._slicer_matrix
        return (
            2
            * slicer_matrix.T
            @ cell_volumes_sqrt.T
            @ weights_matrix.T
            @ weights_matrix
            @ cell_volumes_sqrt
            @ slicer_matrix
        )

    def hessian_diagonal(self, model: Model) -> npt.NDArray[np.float64]:
        """
        Diagonal of the Hessian.

        Parameters
        ----------
        model : (n_params) array
            Array with model values.
        """
        return self.hessian(model).diagonal()

    @property
    def weights_matrix(self) -> dia_array[np.float64]:
        """
        Diagonal matrix with the square root of regularization weights on cells.
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
    def _volumes_sqrt_matrix(self) -> dia_array[np.float64]:
        """
        Diagonal matrix with the square root of cell volumes.
        """
        cell_volumes = self.mesh.cell_volumes[self.active_cells]
        return diags_array(np.sqrt(cell_volumes))

    @property
    def _slicer_matrix(self) -> dia_array[np.float64]:
        """
        Return ``model_slicer.slicer_matrix`` or just a diagonal array.
        """
        slicer_matrix = (
            self.model_slice.slice_matrix
            if self.model_slice is not None
            else eye_array(self.n_params, dtype=np.float64)
        )
        return slicer_matrix


class Flatness(_MeshBasedRegularization):
    r"""
    Flatness regularization.

    Regularize a weighted norm of a spatial derivative of the model.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
        Mesh to use in the regularization.
    direction : {"x", "y", "z"}
        Direction of the spatial derivative.
    active_cells : (n_cells) array or None, optional
        Array full of bools that indicate the active cells in the mesh. It must have the
        same amount of elements as cells in the mesh.
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
        direction: str,
        *,
        active_cells: npt.NDArray[np.bool] | None = None,
        cell_weights: npt.NDArray | dict[str, npt.NDArray] | None = None,
        reference_model: Model | None = None,
        model_slice: ModelSlice | None = None,
    ):
        self.mesh = mesh
        self.direction = direction
        self.active_cells = (
            active_cells
            if active_cells is not None
            else np.ones(self.mesh.n_cells, dtype=bool)
        )
        # assign model_slice after active_cells so n_params is correct during __init__
        self.model_slice = model_slice

        # Assign the cell weights through the setter
        self.cell_weights = (
            cell_weights
            if cell_weights is not None
            else np.ones(self.n_active, dtype=np.float64)
        )

        self.reference_model = (
            reference_model
            if reference_model is not None
            else np.zeros(self.n_params, dtype=np.float64)
        )
        self.set_name(direction)

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
        cell_volumes_sqrt = self._volumes_sqrt_matrix
        cell_gradient = self._cell_gradient
        slicer_matrix = self._slicer_matrix
        return (
            model_diff.T
            @ slicer_matrix.T
            @ cell_gradient.T
            @ cell_volumes_sqrt.T
            @ weights_matrix.T
            @ weights_matrix
            @ cell_volumes_sqrt
            @ cell_gradient
            @ slicer_matrix
            @ model_diff
        )

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
        cell_volumes_sqrt = self._volumes_sqrt_matrix
        cell_gradient = self._cell_gradient
        slicer_matrix = self._slicer_matrix
        return (
            2
            * slicer_matrix.T
            @ cell_gradient.T
            @ cell_volumes_sqrt.T
            @ weights_matrix.T
            @ weights_matrix
            @ cell_volumes_sqrt
            @ cell_gradient
            @ slicer_matrix
            @ model_diff
        )

    def hessian(self, model: Model):  # noqa: ARG002
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
        slicer_matrix = self._slicer_matrix
        return (
            2
            * slicer_matrix.T
            @ cell_gradient.T
            @ cell_volumes_sqrt.T
            @ weights_matrix.T
            @ weights_matrix
            @ cell_volumes_sqrt
            @ cell_gradient
            @ slicer_matrix
        )

    def hessian_diagonal(self, model: Model) -> npt.NDArray[np.float64]:
        """
        Diagonal of the Hessian.

        Parameters
        ----------
        model : (n_params) array
            Array with model values.
        """
        return self.hessian(model).diagonal()

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

    @property
    def _slicer_matrix(self) -> dia_array[np.float64]:
        """
        Return ``model_slicer.slicer_matrix`` or just a diagonal array.
        """
        slicer_matrix = (
            self.model_slice.slice_matrix
            if self.model_slice is not None
            else eye_array(self.n_params, dtype=np.float64)
        )
        return slicer_matrix


class SparseSmallness(_MeshBasedRegularization):
    r"""
    Smallness regularization using lp norm.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
        Mesh to use in the regularization.
    norm : float
        Norm used in the regularization (p).
    active_cells : (n_cells) array or None, optional
        Array full of bools that indicate the active cells in the mesh. It must have the
        same amount of elements as cells in the mesh.
    cell_weights : (n_params) array or dict of (n_params) arrays or None, optional
        Array with cell weights.
        For multiple cell weights, pass a dictionary where keys are strings and values
        are the different weights arrays.
        If None, no cell weights are going to be used.
    reference_model : (n_params) array, optional
        Reference model used in the regularization.
    threshold : float, optional
        IRLS threshold. Symbolized with :math:`\epsilon` in
        Fournier and Oldenburg (2019).
    cooling_factor : float, optional
        Factor used to cool down the ``threshold`` when updating the IRLS.
    model_previous : (n_params) array
        Array with previous model in the iterations. This model is used to build the
        ``R`` matrix.
    irls : bool, optional
        Flag to activate or deactivate IRLS. If False, the class would work as an L2
        smallness term. If True, the R matrix will be built using the
        ``model_previous``.
    """

    def __init__(
        self,
        mesh: discretize.base.BaseMesh,
        *,
        norm: float,
        active_cells: npt.NDArray | None = None,
        cell_weights: npt.NDArray | dict[str, npt.NDArray] | None = None,
        reference_model: Model | None = None,
        threshold: float = 1e-8,
        cooling_factor=1.25,
        model_previous: Model | None = None,
        irls=False,
    ):
        self.mesh = mesh
        self.active_cells = (
            active_cells
            if active_cells is not None
            else np.ones(mesh.n_cells, dtype=bool)
        )
        self.norm = norm
        self.irls = irls
        self.model_previous = (
            model_previous if model_previous is not None else np.zeros(self.n_params)
        )

        # Assign the cell weights through the setter
        self.cell_weights = (
            cell_weights
            if cell_weights is not None
            else np.ones(self.n_params, dtype=np.float64)
        )

        self.reference_model = (
            reference_model
            if reference_model is not None
            else np.zeros(self.n_params, dtype=np.float64)
        )
        self.threshold = threshold
        self.cooling_factor = cooling_factor
        self.set_name(f"s(p={self.norm})")

    def activate_irls(self, model_previous: Model):
        r"""
        Activate IRLS.

        Parameters
        ----------
        model_previous : (n_params) array
            Inverted model obtained after the first stage (l2 inversion).

        Notes
        -----
        Activate IRLS on the regularization, assign ``model_previous`` with the
        ``model_previous`` obtained after the first stage (the l2 inversion,
        before IRLS gets activated), and estimate the initial ``threshold`` as:

        .. math::

            \epsilon = \lVert \mathbf{m}_\text{prev} \rVert_\infty =
            \text{max}(|\mathbf{m}_\text{prev}|)

        where :math:`\mathbf{m}_\text{prev}` is the ``model_previous`` argument.

        """
        self.model_previous = model_previous
        self.threshold = np.max(np.abs(model_previous))
        self.irls = True

    def update_irls(self, model: Model):
        """
        Update IRLS parameters.

        Cool down the threshold and assign the ``model`` as the new ``model_previous``
        attribute.
        """
        self.threshold /= self.cooling_factor
        self.model_previous = model

    @property
    def R(self) -> dia_array:
        """
        R matrix to approximate lp norm using Lawson's algorithm.
        """
        if not self.irls:
            return eye_array(self.n_params)
        power = self.norm / 4 - 0.5
        diagonal = (self.model_previous**2 + self.threshold**2) ** power
        return diags_array(diagonal)

    def __call__(self, model: Model) -> float:
        model_diff = model - self.reference_model
        weights_matrix = self.weights_matrix
        cell_volumes_sqrt = self._volumes_sqrt_matrix
        r_matrix = self.R
        return (
            model_diff.T
            @ r_matrix.T
            @ cell_volumes_sqrt.T
            @ weights_matrix.T
            @ weights_matrix
            @ cell_volumes_sqrt
            @ r_matrix
            @ model_diff
        )

    def gradient(self, model: Model):
        """
        Gradient vector.
        """
        model_diff = model - self.reference_model
        weights_matrix = self.weights_matrix
        cell_volumes_sqrt = self._volumes_sqrt_matrix
        r_matrix = self.R
        return (
            2
            * r_matrix.T
            @ cell_volumes_sqrt.T
            @ weights_matrix.T
            @ weights_matrix
            @ cell_volumes_sqrt
            @ r_matrix
            @ model_diff
        )

    def hessian(self, model: Model):  # noqa: ARG002
        """
        Hessian matrix.
        """
        weights_matrix = self.weights_matrix
        r_matrix = self.R
        cell_volumes_sqrt = self._volumes_sqrt_matrix
        return (
            2
            * r_matrix.T
            @ cell_volumes_sqrt.T
            @ weights_matrix.T
            @ weights_matrix
            @ cell_volumes_sqrt
            @ r_matrix
        )

    def hessian_diagonal(self, model: Model) -> npt.NDArray[np.float64]:
        """
        Diagonal of the Hessian.
        """
        return self.hessian(model).diagonal()

    @property
    def weights_matrix(self) -> dia_array:
        """
        Diagonal matrix with the square root of regularization weights on cells.
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
