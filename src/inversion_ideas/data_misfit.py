"""
Class to represent a data misfit term.
"""

from scipy.sparse import diags_array

from .objective_function import Objective


class DataMisfit(Objective):
    """
    L2 data misfit term.
    """

    def __init__(self, data, uncertainty, simulation):
        self.data = data
        self.uncertainty = uncertainty
        self.simulation = simulation

    def __call__(self, model) -> float:
        residual = self.residual(model)
        return residual.T @ self.weights_squared @ residual

    def gradient(self, model):
        """
        Gradient vector.
        """
        jac = self.simulation.jacobian(model)
        return -2 * jac.T @ self.weights_squared @ self.residual(model)

    def hessian(self, model):
        """
        Hessian matrix.
        """
        jac = self.simulation.jacobian(model)
        return 2 * jac.T @ self.weights_squared @ jac

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
        return diags_array(1 / self.uncertainty)
