"""
Inversion framework to implement a linear regressor.
"""
import time
import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator

from inversion_ideas.base import Simulation
from inversion_ideas.utils import cache_on_model


class LinearRegressor(Simulation):
    r"""
    Linear regressor.

    .. math::

        \mathbf{y} = \mathbf{X} \cdot \mathbf{m}
    """

    def __init__(self, X, linop=False, sleep=0, cache=True):
        self.X = X
        self.linop = linop
        self.sleep = sleep
        self.cache = cache

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

    @cache_on_model
    def __call__(self, model) -> NDArray[np.float64]:
        """
        Evaluate simulation for a given model.
        """
        if self.sleep != 0:
            time.sleep(self.sleep)
        return self.X @ model

    def jacobian(self, model) -> NDArray[np.float64] | LinearOperator:  # noqa: ARG002
        """
        Jacobian matrix for a given model.
        """
        if self.linop:
            linear_op = LinearOperator(
                shape=(self.n_data, self.n_params),
                matvec=lambda model: self.X @ model,
                rmatvec=lambda model: self.X.T @ model,
                dtype=np.float64,
            )
            return linear_op
        return self.X
