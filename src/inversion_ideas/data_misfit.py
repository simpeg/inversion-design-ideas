"""
Class to represent a data misfit term.
"""

import numpy as np
import numpy.typing as npt
from scipy.sparse import diags_array, sparray
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from .base import Objective
from .utils import cache_on_model


class DataMisfit(Objective):
    """
    L2 data misfit term.

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
    """

    def __init__(
        self, data, uncertainty, simulation, *, cache=False, build_hessian=False
    ):
        self.data = data
        self.uncertainty = uncertainty
        self.simulation = simulation
        self.cache = cache
        self.build_hessian = build_hessian
        self.set_name("d")

    @cache_on_model
    def __call__(self, model) -> float:
        residual = self.residual(model)
        return residual.T @ self.weights_squared @ residual

    def gradient(self, model) -> npt.NDArray[np.float64]:
        """
        Gradient vector.
        """
        jac = self.simulation.jacobian(model)
        return -2 * jac.T @ (self.weights_squared @ self.residual(model))

    def hessian(self, model) -> npt.NDArray[np.float64] | sparray | LinearOperator:
        """
        Hessian matrix.
        """
        jac = self.simulation.jacobian(model)
        if not self.build_hessian:
            jac = aslinearoperator(jac)
        return 2 * jac.T @ aslinearoperator(self.weights_squared) @ jac

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
    def weights_squared(self):
        """
        Diagonal sparse matrix with weights squared.
        """
        # Return the W.T @ W matrix
        return diags_array(1 / self.uncertainty**2)
