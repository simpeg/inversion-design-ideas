"""
Test utility functions.
"""

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import csr_array, csr_matrix, diags_array, eye_array, sparray
from scipy.sparse.linalg import aslinearoperator

from inversion_ideas.base import Objective
from inversion_ideas.errors import ConvergenceError
from inversion_ideas.typing import SparseArray
from inversion_ideas.utils import (
    estimate_tradeoff_parameter,
    get_largest_eigenvalue,
    get_sensitivity_weights,
)


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


def get_largest_eigenvalue_numpy(matrix):
    """
    Get the largest absolute eigenvalue of a matrix using :func:`numpy.linalg.ein`.

    Parameters
    ----------
    matrix : (n, n) array
        Dense 2D array.

    Returns
    -------
    float
    """
    eigenvalues, _ = np.linalg.eig(matrix)
    eigenvalues = np.abs(eigenvalues)
    eigenvalues.sort()
    return eigenvalues[-1]


class TestLargestEigenvalue:
    """Test the ``get_largest_eigenvalue`` function."""

    n = 5

    @pytest.fixture
    def dense_matrix(self):
        """Sample dense matrix."""
        rng = np.random.default_rng(seed=42)
        matrix = rng.random(size=(self.n, self.n))
        return matrix

    @pytest.fixture(params=["dense", "sparse", "linop"])
    def matrix(self, dense_matrix, request):
        """Sample matrix argument for the ``get_largest_eigenvalue`` function."""
        match request.param:
            case "dense":
                return dense_matrix
            case "sparse":
                return csr_array(dense_matrix)
            case "linop":
                return aslinearoperator(dense_matrix)
            case _:  # pragma: nocover
                raise ValueError()

    def test_largest_eigenvalue(self, matrix, dense_matrix):
        """Compare estimated eigenvalue with true one computed using Numpy."""
        # Estimate absolute largest eigenvalue
        eigenvalue = get_largest_eigenvalue(matrix, seed=42)

        # Get exact eigenvalues
        expected = get_largest_eigenvalue_numpy(dense_matrix)

        # Compare estimated with desired
        np.testing.assert_allclose(eigenvalue, expected, rtol=5e-4)

    def test_null_vector_error(self):
        """Check that null vector error is raised."""
        matrix = np.zeros((self.n, self.n))
        with pytest.raises(ConvergenceError, match="null vector"):
            get_largest_eigenvalue(matrix)

    def test_invalid_matrix(self):
        matrix = np.zeros(self.n)
        with pytest.raises(ValueError, match="It must be a 2D dense or sparse array"):
            get_largest_eigenvalue(matrix)

    def test_non_square_matrix(self):
        matrix = np.zeros((5, 4))
        with pytest.raises(ValueError, match="Invalid non-square matrix"):
            get_largest_eigenvalue(matrix)


class DummyWithMatrix(Objective):
    """
    Dummy objective function class used to test ``estimate_tradeoff_parameter``.
    """

    def __init__(self, n_params: int, seed=None, hessian_type="dense"):
        self._n_params = n_params
        if hessian_type not in ("dense", "sparse", "linop"):
            raise ValueError()  # pragma: nocover
        self.hessian_type = hessian_type
        shape = (n_params, n_params)
        self.hessian_matrix = np.random.default_rng(seed=seed).random(size=shape)

    @property
    def n_params(self):
        return self._n_params

    def __call__(self, model):
        raise NotImplementedError()  # pragma: nocover

    def gradient(self, model):
        raise NotImplementedError()  # pragma: nocover

    def hessian(self, model):  # ruff: ignore[ARG002]
        match self.hessian_type:
            case "dense":
                return self.hessian_matrix
            case "sparse":
                return csr_matrix(self.hessian_matrix)
            case "linop":
                return aslinearoperator(self.hessian_matrix)
            case _:  # pragma: nocover
                raise ValueError()


class TestEstimateTradeOffParameter:
    """
    Test the ``estimate_tradeoff_parameter``.
    """

    n_params = 15

    @pytest.mark.parametrize("hessian_one_type", ["dense", "sparse", "linop"])
    @pytest.mark.parametrize("hessian_two_type", ["dense", "sparse", "linop"])
    def test_estimate_tradeoff_parameter(self, hessian_one_type, hessian_two_type):
        """Test parameter estimation by comparing against true eigenvalues."""
        phi_1 = DummyWithMatrix(self.n_params, seed=42, hessian_type=hessian_one_type)
        phi_2 = DummyWithMatrix(self.n_params, seed=43, hessian_type=hessian_two_type)

        # Use function to estimate trade-off parameter
        model = np.random.default_rng(seed=4141).random(size=self.n_params)
        beta = estimate_tradeoff_parameter(phi_1, phi_2, model, seed=41)

        # Get the expected value by computing exact eigenvalues of the dense matrices
        eigen_1 = get_largest_eigenvalue_numpy(phi_1.hessian_matrix)
        eigen_2 = get_largest_eigenvalue_numpy(phi_2.hessian_matrix)
        expected = eigen_1 / eigen_2

        np.testing.assert_allclose(beta, expected, rtol=1e-5)

    def test_invalid_objective(self):
        """Test error after incompatible objective functions."""
        phi_1 = DummyWithMatrix(10)
        phi_2 = DummyWithMatrix(11)
        model = np.random.default_rng(seed=4141).random(size=phi_1.n_params)
        with pytest.raises(ValueError, match="Invalid objective functions"):
            estimate_tradeoff_parameter(phi_1, phi_2, model)
