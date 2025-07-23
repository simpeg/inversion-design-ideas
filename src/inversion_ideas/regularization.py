"""
Regularization classes.
"""
import numpy as np
from scipy.sparse import diags_array

from .objective_function import Objective


class TikhonovZero(Objective):
    """
    Tikhonov zero-th order regularization.
    """

    def __init__(self, n_params: int, weights=None, reference_model=None):
        self._n_params = n_params
        self.weights = (
            weights if weights is not None else np.ones(n_params, dtype=np.float64)
        )
        self.reference_model = (
            reference_model
            if reference_model is not None
            else np.zeros(n_params, dtype=np.float64)
        )

    def __call__(self, model) -> float:
        model_diff = model - self.reference_model
        weights = diags_array(self.weights)
        return model_diff.T @ weights.T @ weights @ model_diff

    def gradient(self, model):
        """
        Gradient vector.
        """
        model_diff = model - self.reference_model
        weights = diags_array(self.weights)
        return 2 * weights.T @ weights @ model_diff

    def hessian(self, model):  # noqa: ARG002
        """
        Hessian matrix.
        """
        weights = diags_array(self.weights)
        return 2 * weights.T @ weights

    @property
    def n_params(self):
        """
        Number of model parameters.
        """
        return self._n_params
