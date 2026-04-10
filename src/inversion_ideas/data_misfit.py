"""
Class to represent a data misfit term.
"""

import numpy as np
import numpy.typing as npt
from scipy.sparse import dia_array, diags_array
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from .base import Objective
from .operators import get_diagonal
from .typing import Model, SparseArray


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
        build_hessian=False,
    ):
        # TODO: Check that the data and uncertainties have the size as ndata in the
        #       simulation.
        self.data = data
        self.uncertainty = uncertainty
        self.simulation = simulation
        self.build_hessian = build_hessian
        self.set_name("d")

    def __call__(self, model: Model) -> float:
        residual = self.residual(model)
        weights_matrix = self.weights_matrix
        return residual.T @ weights_matrix.T @ weights_matrix @ residual

    def gradient(self, model: Model) -> npt.NDArray[np.float64]:
        """
        Gradient vector.
        """
        jac = self.simulation.jacobian(model)
        weights_matrix = self.weights_matrix
        return 2 * jac.T @ (weights_matrix.T @ weights_matrix @ self.residual(model))

    def hessian(
        self, model: Model
    ) -> npt.NDArray[np.float64] | SparseArray | LinearOperator:
        """
        Hessian matrix.
        """
        jac = self.simulation.jacobian(model)

        if self.build_hessian and isinstance(jac, LinearOperator):
            msg = (
                f"Cannot build Hessian for DataMisfit '{self}' since the Jacobian "
                f"of {self.simulation} is a LinearOperator. "
                f"Set `build_hessian` to False in '{self}', or adjust your "
                "simulation to return a dense or sparse Jacobian matrix."
            )
            raise TypeError(msg)

        if not self.build_hessian:
            jac = aslinearoperator(jac)
        weights_matrix = aslinearoperator(self.weights_matrix)
        return 2 * jac.T @ weights_matrix.T @ weights_matrix @ jac

    def hessian_approx(self, model: Model) -> npt.NDArray[np.float64] | SparseArray:
        """
        Approximated version of the Hessian.

        If ``build_hessian`` is True, then the full Hessian will be returned.
        Otherwise, the Hessian will be approximated by a diagonal sparse matrix, whose
        main diagonal matches Hessian's diagonal.

        Parameters
        ----------
        model : (n_params) array
            Array with model values.

        Returns
        -------
        (n_params, n_params) dense or sparse array
            2D diagonal dense or sparse array that approximates the Hessian of the
            objective function.
        """
        if self.build_hessian:
            # Ignore type error: if build_hessian is True, then hessian(model) will
            # always return a dense or sparse array.
            return self.hessian(model)  # type: ignore[return-value]

        jac = self.simulation.jacobian(model)
        if isinstance(jac, LinearOperator):
            # Repeat hessian implementation here to avoid recomputing the jacobian
            weights_matrix = aslinearoperator(self.weights_matrix)
            hessian = 2 * jac.T @ weights_matrix.T @ weights_matrix @ jac
            # Estimate diagonal
            diagonal = get_diagonal(hessian)
        else:
            diagonal = 2 * np.einsum("i,ij,ij->j", self.weights, jac, jac)
        return diags_array(diagonal)

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
        return self.simulation(model) - self.data

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
