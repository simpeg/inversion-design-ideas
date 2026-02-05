"""
Class to represent a data misfit term.
"""

import numpy as np
import numpy.typing as npt
from scipy.sparse import dia_array, diags_array, eye_array, sparray
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from .base import Objective
from .typing import Model
from .utils import cache_on_model
from .wires import ModelSlice


class DataMisfit(Objective):
    r"""
    L2 data misfit.

    Parameters
    ----------
    data : (n_data) array
        Array with observed data values.
    uncertainty : (n_data) array
        Array with data uncertainty.
    simulation : Simulation
        Instance of Simulation.
    cache : bool, optional
        Whether to cache the last result of the `__call__` method.
        Default to False.
    build_hessian : bool, optional
        If True, the ``hessian`` method will build the Hessian matrix and allocate it in
        memory. If False, the ``hessian`` method will return a linear operator that
        represents the Hessian matrix. Default to False.

        .. important::

            Hessian matrices are usually very large. Use ``build_hessian=True`` only if
            you need to build it.

    Notes
    -----
    The L2 data misfit objective function is defined as:

    .. math::

        \phi_d(\mathbf{m}) =
        \sum\limits_{i=1}^N
        \frac{\lvert d_i^\text{obs} - f_i(\mathbf{m}) \rvert^2}{\epsilon_i^2}

    where :math:`\mathbf{m}` is the model vector, :math:`d_i^\text{obs}` is the
    :math:`i`-th observed datum, :math:`f_i(\mathbf{m})` is the forward modelling
    function for the :math:`i`-th datum, and :math:`\epsilon_i` is the uncertainty of
    the :math:`i`-th datum.

    The data misfit term can be expressed in terms of weights :math:`w_i
    = 1 / \epsilon_i^2`:

    .. math::

        \phi_d(\mathbf{m}) =
        \sum\limits_{i=1}^N
        w_i \lvert d_i^\text{obs} - f_i(\mathbf{m}) \rvert^2

    And also in matrix form:

    .. math::

        \phi_d(\mathbf{m}) =
        \lVert
        \mathbf{W} \left[ \mathbf{d}^\text{obs} - f(\mathbf{m}) \right]
        \rVert^2

    where :math:`\mathbf{W}` is a diagonal matrix with the square root of the weights,
    :math:`\mathbf{d}^\text{obs}` is the vector of observed data, and
    :math:`f(\mathbf{m})` is the forward modelling vector.

    """

    def __init__(
        self,
        data: npt.NDArray[np.float64],
        uncertainty: npt.NDArray[np.float64],
        simulation,
        *,
        model_slice: ModelSlice | None = None,
        cache=False,
        build_hessian=False,
    ):
        # TODO: Check that the data and uncertainties have the size as ndata in the
        #       simulation.
        self.data = data
        self.uncertainty = uncertainty
        self.simulation = simulation
        self.model_slice = model_slice
        self.cache = cache
        self.build_hessian = build_hessian
        self.set_name("d")

    @cache_on_model
    def __call__(self, model: Model) -> float:
        # TODO:
        # Cache invalidation: we should clean the cache if data or uncertainties change.
        # Or they should be immutable.
        residual = self.residual(model)
        weights_matrix = self.weights_matrix
        return residual.T @ weights_matrix.T @ weights_matrix @ residual

    def gradient(self, model: Model) -> npt.NDArray[np.float64]:
        """
        Gradient vector.
        """
        jac = self._get_jacobian(model)
        weights_matrix = self.weights_matrix
        gradient = (
            2
            * self._slicer_matrix.T
            @ jac.T
            @ (weights_matrix.T @ weights_matrix @ self.residual(model))
        )
        return gradient

    def hessian(
        self, model: Model
    ) -> npt.NDArray[np.float64] | sparray | LinearOperator:
        """
        Hessian matrix.
        """
        jac = self._get_jacobian(model)
        weights_matrix = aslinearoperator(self.weights_matrix)
        slicer_matrix = aslinearoperator(self._slicer_matrix)
        if not self.build_hessian:
            jac = aslinearoperator(jac)
        return (
            2
            * slicer_matrix.T
            @ jac.T
            @ weights_matrix.T
            @ weights_matrix
            @ jac
            @ slicer_matrix
        )

    def hessian_diagonal(self, model: Model) -> npt.NDArray[np.float64]:
        """
        Approximated diagonal of the Hessian.
        """
        jac = self._get_jacobian(model)
        if isinstance(jac, LinearOperator):
            msg = (
                "`DataMisfit.hessian_diagonal()` is not implemented for simulations "
                "that return the jacobian as a LinearOperator."
            )
            raise NotImplementedError(msg)

        jtj_diag = np.einsum("i,ij,ij->j", self.weights_matrix.diagonal(), jac, jac)
        return 2 * self._slicer_matrix.T @ jtj_diag

    @property
    def n_params(self):
        """
        Number of model parameters.
        """
        if self.model_slice is not None:
            return self.model_slice.full_size
        return self.simulation.n_params

    @property
    def n_data(self):
        """
        Number of data values.
        """
        return self.data.size

    def residual(self, model: Model):
        r"""
        Residual vector.

        Parameters
        ----------
        model : (n_params) array
            Array with model values.

        Returns
        -------
        (n_data) array
            Array with residual vector.

        Notes
        -----
        Residual vector defined as:

        .. math::

            \mathbf{r} = \mathcal{F}(\mathbf{m}) - \mathbf{d}

        where :math:`\mathbf{d}` is the vector with observed data, :math:`\mathcal{F}`
        is the forward model, and :math:`\mathbf{m}` is the model vector.
        """
        if self.model_slice is not None:
            model = self.model_slice.extract(model)
        return self.simulation(model) - self.data

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """
        Data weights: 1D array with the square of the inverse of the uncertainties.
        """
        return 1 / self.uncertainty**2

    @property
    def weights_matrix(self) -> dia_array[np.float64]:
        """
        Diagonal matrix with the square root of the regularization weights.
        """
        return diags_array(1 / self.uncertainty)

    def chi_factor(self, model: Model):
        """
        Compute chi factor.

        Parameters
        ----------
        model : (n_params) array
            Array with model values.

        Return
        ------
        float
            Chi factor for the given model.
        """
        return self(model) / self.n_data

    def _get_jacobian(self, model: Model) -> LinearOperator:
        """
        Return the jacobian of the simulation evaluated on the passed model.

        Parameters
        ----------
        model : (n_params) array
            Model array. The array must have the full size. This method will use the
            ``model_slice`` to extract the relevant portion that will be passed to the
            simulation.
        """
        jac = self.simulation.jacobian(
            model if self.model_slice is None else self.model_slice.extract(model)
        )
        return jac

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
