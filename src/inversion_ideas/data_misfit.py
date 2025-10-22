"""
Class to represent a data misfit term.
"""

import numpy as np
import numpy.typing as npt
from scipy.sparse import dia_array, diags_array, sparray
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from .base import Objective
from .utils import cache_on_model


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
        cache=False,
        build_hessian=False,
    ):
        self.data = data
        self.uncertainty = uncertainty
        self.simulation = simulation
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
        jac = self.simulation.jacobian(model)
        weights_matrix = self.weights_matrix
        return -2 * jac.T @ (weights_matrix.T @ weights_matrix @ self.residual(model))

    def hessian(
        self, model: Model
    ) -> npt.NDArray[np.float64] | sparray | LinearOperator:
        """
        Hessian matrix.
        """
        jac = self.simulation.jacobian(model)
        if not self.build_hessian:
            jac = aslinearoperator(jac)
        weights_matrix = aslinearoperator(self.weights_matrix)
        return 2 * jac.T @ weights_matrix.T @ weights_matrix @ jac

    def hessian_diagonal(self, model) -> npt.NDArray[np.float64]:
        """
        Diagonal of the Hessian.
        """
        jac = self.simulation.jacobian(model)
        if isinstance(jac, LinearOperator):
            msg = (
                "`DataMisfit.hessian_diagonal()` is not implemented for simulations "
                "that return the jacobian as a LinearOperator."
            )
            raise NotImplementedError(msg)
        jtj_diag = np.einsum("i,ij,ij->j", self.weights_matrix.diagonal(), jac, jac)
        return 2 * jtj_diag

    @property
    def n_params(self):
        """
        Number of model parameters.
        """
        return self.simulation.n_params

    @property
    def n_data(self):
        """
        Number of data values.
        """
        return self.data.size

    def residual(self, model):
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

            \mathbf{r} = \mathbf{d} - \mathcal{F}(\mathbf{m})

        where :math:`\mathbf{d}` is the vector with observed data, :math:`\mathcal{F}`
        is the forward model, and :math:`\mathbf{m}` is the model vector.
        """
        return self.data - self.simulation(model)

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """
        Data weights: 1D array with the square of the inverse of the uncertainties.
        """
        return 1 / self.uncertainty**2

    @property
    def weights_matrix(self) -> dia_array:
        """
        Diagonal matrix with the square root of the regularization weights.
        """
        return diags_array(1 / self.uncertainty)

    def chi_factor(self, model):
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
