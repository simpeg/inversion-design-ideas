"""
Regularization classes.
"""

import numpy as np
from numpy import typing as npt
from scipy.sparse import diags_array, eye_array

from .base import Objective


class TikhonovZero(Objective):
    """
    Tikhonov zero-th order regularization.

    Parameters
    ----------
    n_params : int
        Number of elements of the model vector.
    weights : (n_params) array, optional
        Array with weights.
    reference_model : (n_params) array, optional
        Reference model used in the regularization.
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
        self.set_name("s")

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

    def hessian_diagonal(self, model) -> npt.NDArray[np.float64]:
        """
        Diagonal of the Hessian.
        """
        return self.hessian(model).diagonal()

    @property
    def n_params(self):
        """
        Number of model parameters.
        """
        return self._n_params


class SparseSmallness(Objective):
    r"""
    Smallness regularization using lp norm.

    Parameters
    ----------
    n_params : int
        Number of elements of the model vector.
    norm : float
        Norm used in the regularization (p).
    model_previous : (n_params) array
        Array with previous model in the iterations. This model is used to build the
        ``R`` matrix.
    irls : bool, optional
        Flag to activate or deactivate IRLS. If False, the class would work as an L2
        smallness term. If True, the R matrix will be built using the
        ``model_previous``.
    weights : (n_params) array, optional
        Array with weights.
    reference_model : (n_params) array, optional
        Reference model used in the regularization.
    threshold : float, optional
        IRLS threshold. Symbolized with :math:`\epsilon` in
        Fournier and Oldenburg (2019).
    cooling_factor : float, optional
        Factor used to cool down the ``threshold`` when updating the IRLS.
    """

    def __init__(
        self,
        n_params: int,
        norm: float,
        model_previous: npt.NDArray,
        irls=False,
        weights=None,
        reference_model=None,
        threshold: float = 1e-8,
        cooling_factor=1.25,
    ):
        self._n_params = n_params
        self.norm = norm
        self.irls = irls
        self.model_previous = model_previous
        self.weights = (
            weights if weights is not None else np.ones(n_params, dtype=np.float64)
        )
        self.reference_model = (
            reference_model
            if reference_model is not None
            else np.zeros(n_params, dtype=np.float64)
        )
        self.threshold = threshold
        self.cooling_factor = cooling_factor
        self.set_name("ss")

    def activate_irls(self, model_previous):
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

    def update_irls(self, model):
        """
        Update IRLS parameters.

        Cool down the threshold and assign the ``model`` as the new ``model_previous``
        attribute.
        """
        self.threshold /= self.cooling_factor
        self.model_previous = model

    @property
    def R(self):
        """
        R matrix to approximate lp norm using Lawson's algorithm.
        """
        if not self.irls:
            return eye_array(self.n_params)
        power = self.norm / 4 - 0.5
        diagonal = (self.model_previous**2 + self.threshold**2) ** power
        return diags_array(diagonal)

    def __call__(self, model) -> float:
        model_diff = model - self.reference_model
        weights = diags_array(self.weights)
        r_matrix = self.R
        return model_diff.T @ r_matrix.T @ weights.T @ weights @ r_matrix @ model_diff

    def gradient(self, model):
        """
        Gradient vector.
        """
        model_diff = model - self.reference_model
        weights = diags_array(self.weights)
        r_matrix = self.R
        return 2 * r_matrix.T @ weights.T @ weights @ r_matrix @ model_diff

    def hessian(self, model):  # noqa: ARG002
        """
        Hessian matrix.
        """
        weights = diags_array(self.weights)
        r_matrix = self.R
        return 2 * r_matrix.T @ weights.T @ weights @ r_matrix

    @property
    def n_params(self):
        """
        Number of model parameters.
        """
        return self._n_params
