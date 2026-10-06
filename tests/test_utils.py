"""
Test utility functions.
"""

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import csr_array, diags_array, eye_array, sparray
from scipy.sparse.linalg import aslinearoperator

from inversion_ideas.typing import SparseArray
from inversion_ideas.utils import get_sensitivity_weights


def dumb_sensitivity_weights(
    jacobian: NDArray[np.float64] | SparseArray,
    data_weights: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """
    Simple implementation of sensitivity weights just for testing.
    """
    # Build weights matrix as a diagonal matrix with sqrt roots of the weights
    if data_weights is None:
        weights_matrix = eye_array(jacobian.shape[0])
    else:
        weights_matrix = diags_array(np.sqrt(data_weights))

    # Get dense jacobian if it's a sparse array
    if isinstance(jacobian, sparray):
        jacobian = jacobian.toarray()

    sensitivity_weights = np.sqrt(np.sum((weights_matrix @ jacobian) ** 2, axis=0))

    # Normalize them
    sensitivity_weights /= sensitivity_weights.max()
    return sensitivity_weights


class TestSensitivityWeights:
    """
    Test the ``get_sensitivity_weights`` function.
    """

    shape = (5, 3)

    @pytest.fixture(params=["dense", "sparse"])
    def jacobian(self, request):
        """
        Generate a random jacobian matrix.
        """
        jacobian = np.random.default_rng(seed=9894).uniform(size=self.shape)
        if request.param == "dense":
            return jacobian
        if request.param == "sparse":
            return csr_array(jacobian)
        raise ValueError()  # pragma: nocover

    @pytest.fixture(params=[None, "array"])
    def data_weights(self, request):
        if request.param is None:
            return None
        if request.param == "array":
            return np.random.default_rng(seed=14124).uniform(size=self.shape[0])
        raise ValueError()  # pragma: nocover

    def test_sensitivity_weights(self, jacobian, data_weights):
        """Test if sensitivity weights are correctly computed."""
        # Compute sensitivity weights for different combinations of jacobian types
        # (dense or sparse 2D array) and data_weights types (None or 1D array).
        sensitivity_weights = get_sensitivity_weights(
            jacobian, data_weights=data_weights
        )
        np.testing.assert_allclose(
            sensitivity_weights, dumb_sensitivity_weights(jacobian, data_weights)
        )

    def test_invalid_jacobian(self):
        jacobian_linop = aslinearoperator(
            np.random.default_rng(seed=1414).uniform(size=self.shape)
        )
        with pytest.raises(TypeError, match="Invalid jacobian"):
            get_sensitivity_weights(jacobian_linop)

    def test_invalid_data_weights(self, jacobian):
        data_weights = np.random.default_rng(seed=14142).uniform(size=self.shape)
        with pytest.raises(
            ValueError, match="Invalid data_weights array with '2' dimensions"
        ):
            get_sensitivity_weights(jacobian, data_weights=data_weights)
