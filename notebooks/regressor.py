"""
Inversion framework to implement a linear regressor.
"""
import numpy as np
from numpy.typing import NDArray

from inversion_ideas import Simulation


class LinearRegressor(Simulation):
    r"""
    Linear regressor.

    .. math::

        \mathbf{y} = \mathbf{X} \cdot \mathbf{m}
    """

    def __init__(self, X):
        self.X = X

    @property
    def n_params(self) -> int:
        """
        Number of model parameters.
        """
        return self.X.shape[1]

    @property
    def n_data(self) -> int:
        """
        Number of data values.
        """
        return self.X.shape[0]

    def __call__(self, model) -> NDArray[np.float64]:
        """
        Evaluate simulation for a given model.
        """
        return self.X @ model

    def jacobian(self, model):  # noqa: ARG002
        """
        Jacobian matrix for a given model.
        """
        return self.X
